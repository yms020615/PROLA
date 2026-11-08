from datetime import timedelta

import torch
from einops import rearrange
from torch import nn

from prola.models._aurora.batch import Batch
from prola.models._aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)
from prola.models._aurora.model.patchembed import LevelPatchEmbed
from prola.models._aurora.model.perceiver import MLP, PerceiverResampler
from prola.models._aurora.model.posencoding import pos_scale_enc
from prola.models._aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(nn.Module):

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        stabilise_level_agg: bool = False,
    ) -> None:

        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels

        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(
            surf_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )
        self.atmos_token_embeds = LevelPatchEmbed(
            atmos_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )

        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:

        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)

        x = self.level_agg(latents, x)
        x = x.unflatten(dim=0, sizes=(B, L))
        x = torch.einsum("blcd->bcld", x)
        return x

    def forward(self, batch: Batch, lead_time: timedelta) -> torch.Tensor:

        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)
        x_static = torch.stack(tuple(batch.static_vars.values()), dim=2)
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)

        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        if static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)
        dtype = x_surf.dtype

        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)

        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed

        x_atmos = self.aggregate_levels(x_atmos)

        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )

        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        x = x.reshape(B, -1, self.embed_dim)

        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        x = x + lead_time_emb.unsqueeze(1)

        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)

        x = self.pos_drop(x)
        return x
