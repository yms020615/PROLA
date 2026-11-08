import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

__all__ = ["LevelPatchEmbed"]


class LevelPatchEmbed(nn.Module):

    def __init__(
        self,
        var_names: tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        history_size: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:

        super().__init__()

        self.var_names = var_names
        self.kernel_size = (history_size,) + to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size)) for name in var_names}
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:

        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: tuple[str, ...]) -> torch.Tensor:

        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[0]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[1]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        weight = torch.cat(
            [self.weights[name][:, :, :T, ...] for name in var_names],
            dim=1,
        )

        stride = (T,) + self.kernel_size[1:]

        proj = F.conv3d(x, weight, self.bias, stride=stride)
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)
            proj = proj.transpose(1, 2)

        x = self.norm(proj)
        return x
