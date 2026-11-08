from datetime import timedelta

import torch
from einops import rearrange
from torch import nn

from prola.models._aurora.batch import Batch
from prola.models._aurora.model.fourier import levels_expansion
from prola.models._aurora.model.perceiver import PerceiverResampler
from prola.models._aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)

__all__ = ["Perceiver3DDecoder"]


class Perceiver3DDecoder(nn.Module):

    def __init__(
        self,
        out_surf_vars: tuple[str, ...],
        out_atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
    ) -> None:

        super().__init__()

        self.patch_size = patch_size

        self.embed_dim = embed_dim
        self.out_surf_vars = out_surf_vars
        self.out_atmos_vars = out_atmos_vars

        if out_surf_vars:
            self.surf_heads = nn.ParameterDict(
                {name: nn.Linear(embed_dim, patch_size**2) for name in out_surf_vars}
            )

        if out_atmos_vars:
            self.atmos_heads = nn.ParameterDict(
                {name: nn.Linear(embed_dim, patch_size**2) for name in out_atmos_vars}
            )

            self.level_decoder = PerceiverResampler(
                latent_dim=embed_dim,
                context_dim=embed_dim,
                depth=depth,
                head_dim=head_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                residual_latent=True,
                ln_eps=perceiver_ln_eps,
            )

            self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)
        x = x.flatten(0, 1)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert len(level_embed.shape) == 3, f"Expected 3 dims, found {level_embed.dims()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."

        x = self.level_decoder(level_embed, x)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        lead_time: timedelta,
    ) -> Batch:

        p = next(self.parameters())
        if x.dtype != p.dtype:
            x = x.to(p.dtype)

        surf_vars = self.out_surf_vars
        atmos_vars = self.out_atmos_vars
        atmos_levels = batch.metadata.atmos_levels

        B, L, D = x.shape

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )
        surf_preds = None
        atmos_preds = None

        if surf_vars:

            x_surf = torch.stack(
                [self.surf_heads[name](x[..., :1, :]) for name in surf_vars], dim=-1
            )
            x_surf = x_surf.reshape(*x_surf.shape[:3], -1)
            surf_preds = unpatchify(x_surf, len(surf_vars), H, W, self.patch_size)
            surf_preds = surf_preds.squeeze(2)

        if atmos_vars:

            atmos_levels_encode = levels_expansion(
                torch.tensor(atmos_levels, device=x.device), self.embed_dim
            ).to(dtype=x.dtype)
            levels_embed = self.atmos_levels_embed(atmos_levels_encode)

            levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
            x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :])

            x_atmos = torch.stack([self.atmos_heads[name](x_atmos) for name in atmos_vars], dim=-1)
            x_atmos = x_atmos.reshape(*x_atmos.shape[:3], -1)
            atmos_preds = unpatchify(x_atmos, len(atmos_vars), H, W, self.patch_size)

        return surf_preds, atmos_preds
