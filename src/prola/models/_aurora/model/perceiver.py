import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["MLP", "PerceiverResampler"]


class MLP(nn.Module):

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)


class PerceiverAttention(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        ln_k_q: bool = False,
    ) -> None:

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

        if ln_k_q:
            self.ln_k = nn.LayerNorm(num_heads * head_dim)
            self.ln_q = nn.LayerNorm(num_heads * head_dim)
        else:
            self.ln_k = lambda x: x
            self.ln_q = lambda x: x

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        h = self.num_heads

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        k = self.ln_k(k)
        q = self.ln_q(q)

        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
        ln_k_q: bool = False,
    ) -> None:

        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                            ln_k_q=ln_k_q if i == 0 else False,
                        ),
                        MLP(dim=latent_dim, hidden_features=mlp_hidden_dim, dropout=drop),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        for attn, ff, ln1, ln2 in self.layers:

            attn_out = ln1(attn(latents, x))

            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents
