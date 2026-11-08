import torch
from torch import nn

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0) -> None:

        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim * 2))
        self.scale_bias = scale_bias

        self.init_weights()

    def init_weights(self) -> None:

        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

        shift, scale = self.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        return self.ln(x) * (self.scale_bias + scale) + shift
