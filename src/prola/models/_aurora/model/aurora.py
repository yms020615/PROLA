import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from einops import rearrange
from prola.models._aurora.batch import Batch
from prola.models._aurora.model.decoder import Perceiver3DDecoder
from prola.models._aurora.model.encoder import Perceiver3DEncoder
from prola.models._aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]


class Aurora(torch.nn.Module):

    def __init__(
        self,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        out_surf_vars: tuple[str, ...] = None,
        out_atmos_vars: tuple[str, ...] = None,
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-05,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
    ) -> None:
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.encoder_depths = encoder_depths
        self.decoder_depths = decoder_depths
        out_surf_vars = surf_vars if out_surf_vars is None else out_surf_vars
        out_atmos_vars = atmos_vars if out_atmos_vars is None else out_atmos_vars
        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually adjusted: {', '.join(sorted(self.surf_stats.keys()))}. Please ensure that this is right!",
                stacklevel=2,
            )
        self.encoder = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
        )
        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
        )
        self.decoder = Perceiver3DDecoder(
            out_surf_vars=out_surf_vars,
            out_atmos_vars=out_atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )

    def forward(self, batch: Batch = None) -> Batch:
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)
        (H, W) = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )
        (B, T) = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={
                k: v[None, None].repeat(B, T, 1, 1) for (k, v) in batch.static_vars.items()
            },
        )
        x = self.encoder(batch, lead_time=self.timestep)
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.autocast and p.device.type == "cuda"
            else contextlib.nullcontext()
        ):
            x = self.backbone(
                x,
                lead_time=self.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        dec_dtype = next(self.decoder.parameters()).dtype
        if x.dtype != dec_dtype:
            x = x.to(dec_dtype)
        (surf_preds, atmos_preds) = self.decoder(
            x, batch, lead_time=self.timestep, patch_res=patch_res
        )
        return (surf_preds, atmos_preds)

    def load_checkpoint(self, repo: str, name: str, strict: bool = True) -> None:
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        device = next(self.parameters()).device
        d = torch.load(path, map_location=device, weights_only=True, mmap=True)
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]
            if weight.shape[-1] != self.patch_size:
                print("interpolate surf encoder for changing patch size")
                weight = torch.nn.functional.interpolate(
                    weight,
                    size=weight.shape[2:3] + (self.patch_size, self.patch_size),
                    mode="trilinear",
                    align_corners=False,
                )
            assert weight.shape[1] == 4 + 3
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                if name == "2t" and "T2M" in self.surf_vars:
                    name = "T2M"
                    print("load 2t encoder weight for T2M")
                elif name == "10u" and "U10M" in self.surf_vars:
                    name = "U10M"
                    print("load 10u encoder weight for U10M")
                elif name == "10v" and "V10M" in self.surf_vars:
                    name = "V10M"
                    print("load 10v encoder weight for V10M")
                elif name == "msl" and "MSL" in self.surf_vars:
                    name = "MSL"
                    print("load msl encoder weight for MSL")
                d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]
        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]
            if weight.shape[-1] != self.patch_size:
                print("interpolate atmos encoder for changing patch size")
                weight = torch.nn.functional.interpolate(
                    weight,
                    size=weight.shape[2:3] + (self.patch_size, self.patch_size),
                    mode="trilinear",
                    align_corners=False,
                )
            assert weight.shape[1] == 5
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if name == "z" and "Z" in self.atmos_vars:
                    name = "Z"
                    print("load z encoder weight for Z")
                elif name == "u" and "U" in self.atmos_vars:
                    name = "U"
                    print("load u encoder weight for U")
                elif name == "v" and "V" in self.atmos_vars:
                    name = "V"
                    print("load v encoder weight for V")
                elif name == "t" and "T" in self.atmos_vars:
                    name = "T"
                    print("load t encoder weight for T")
                elif name == "q" and "Q" in self.atmos_vars:
                    name = "Q"
                    print("load q encoder weight for Q")
                elif name == "q" and "r" in self.atmos_vars:
                    name = "r"
                    print("load q encoder weight for r")
                d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]
        if "decoder.surf_head.weight" in d:
            weight = d["decoder.surf_head.weight"]
            bias = d["decoder.surf_head.bias"]
            del d["decoder.surf_head.weight"]
            del d["decoder.surf_head.bias"]
            if weight.shape[0] != 4 * self.patch_size**2:
                print("interpolate surf decoder for changing patch size")
                weight = weight.reshape(4, 4, 4, -1)
                weight = rearrange(weight, "p1 p2 v d -> v d p1 p2")
                weight = torch.nn.functional.interpolate(
                    weight,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                weight = rearrange(weight, "v d p1 p2 -> p1 p2 v d").reshape(-1, 1024)
                bias = bias.reshape(4, 4, 4, -1)
                bias = rearrange(bias, "p1 p2 v d -> v d p1 p2")
                bias = torch.nn.functional.interpolate(
                    bias,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                bias = rearrange(bias, "v d p1 p2 -> p1 p2 (v d)").reshape(-1)
            assert weight.shape[0] == 4 * self.patch_size**2
            assert bias.shape[0] == 4 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 4, -1)
            bias = bias.reshape(self.patch_size**2, 4)
            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                if name == "2t" and "T2M" in self.surf_vars:
                    name = "T2M"
                    print("load 2t decoder weight for T2M")
                elif name == "10u" and "U10M" in self.surf_vars:
                    name = "U10M"
                    print("load 10u decoder weight for U10M")
                elif name == "10v" and "V10M" in self.surf_vars:
                    name = "V10M"
                    print("load 10v decoder weight for V10M")
                elif name == "msl" and "MSL" in self.surf_vars:
                    name = "MSL"
                    print("load msl decoder weight for MSL")
                d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]
        if "decoder.atmos_head.weight" in d:
            weight = d["decoder.atmos_head.weight"]
            bias = d["decoder.atmos_head.bias"]
            del d["decoder.atmos_head.weight"]
            del d["decoder.atmos_head.bias"]
            if weight.shape[0] != 5 * self.patch_size**2:
                print("interpolate surf decoder for changing patch size")
                weight = weight.reshape(4, 4, 5, -1)
                weight = rearrange(weight, "p1 p2 v d -> v d p1 p2")
                weight = torch.nn.functional.interpolate(
                    weight,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                weight = rearrange(weight, "v d p1 p2 -> p1 p2 v d").reshape(-1, 1024)
                bias = bias.reshape(4, 4, 5, -1)
                bias = rearrange(bias, "p1 p2 v d -> v d p1 p2")
                bias = torch.nn.functional.interpolate(
                    bias,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                bias = rearrange(bias, "v d p1 p2 -> p1 p2 (v d)").reshape(-1)
            assert weight.shape[0] == 5 * self.patch_size**2
            assert bias.shape[0] == 5 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 5, -1)
            bias = bias.reshape(self.patch_size**2, 5)
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if name == "z" and "Z" in self.atmos_vars:
                    name = "Z"
                    print("load z decoder weight for Z")
                elif name == "u" and "U" in self.atmos_vars:
                    name = "U"
                    print("load u decoder weight for U")
                elif name == "v" and "V" in self.atmos_vars:
                    name = "V"
                    print("load v decoder weight for V")
                elif name == "t" and "T" in self.atmos_vars:
                    name = "T"
                    print("load t decoder weight for T")
                elif name == "q" and "Q" in self.atmos_vars:
                    name = "Q"
                    print("load q decoder weight for Q")
                elif name == "q" and "r" in self.atmos_vars:
                    name = "r"
                    print("load q decoder weight for r")
                d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]
        if "encoder.atmos_latents" in d:
            weight = d["encoder.atmos_latents"]
            if self.encoder.atmos_latents.shape[0] != weight.shape[0]:
                print("laten")
        name = "2t"
        if "T2M" in self.surf_vars:
            name = "T2M"
        current_history_size = d[f"encoder.surf_token_embeds.weights.{name}"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} into model with `max_history_size` {self.max_history_size}."
            )
        result = self.load_state_dict(d, strict=strict)
        allowed = ("encoder.surf_token_embeds.weights.tp", "decoder.surf_heads.tp_")
        missing = [key for key in result.missing_keys if not key.startswith(allowed)]
        if missing:
            raise ValueError(f"Pretrained checkpoint is missing model weights: {missing}")

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        print("adapt time")
        for name, weight in list(checkpoint.items()):
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                if not weight.shape[2] <= self.max_history_size:
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} into model with `max_history_size` {self.max_history_size}."
                    )
                (D, _, T, p1, p2) = weight.shape
                new_weight = weight.repeat(1, 1, self.max_history_size // T, 1, 1) / (
                    self.max_history_size // T
                )
                checkpoint[name] = new_weight

    def configure_activation_checkpointing(self):
        non_reentrant_wrapper = partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda x: isinstance(x, BasicLayer3D),
        )

    def get_num_layers(self):
        return sum(self.encoder_depths) + sum(self.decoder_depths)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
)
AuroraHighRes = partial(Aurora, patch_size=10, encoder_depths=(6, 8, 8), decoder_depths=(8, 8, 6))
