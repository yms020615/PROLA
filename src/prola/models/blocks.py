import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSwiGLUBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        dropout: float = 0.0,
        zero_init_out: bool = False,
    ):
        super().__init__()
        self.dw = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pw_in = nn.Conv3d(in_channels, hidden_channels * 2, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.pw_out = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        if zero_init_out:
            nn.init.zeros_(self.pw_out.weight)
            nn.init.zeros_(self.pw_out.bias)

    def forward(self, x):
        w = self.dw.weight
        if x.dtype != w.dtype or x.device != w.device:
            x = x.to(device=w.device, dtype=w.dtype)
        x = self.dw(x)
        x = self.pw_in(x)
        u, v = torch.chunk(x, 2, dim=1)
        x = u * F.silu(v)
        x = self.dropout(x)
        x = self.pw_out(x)
        return x


class DepthwiseSwiGLUStack3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        depth: int = 1,
        dropout: float = 0.0,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >=1, got {depth}")
        blocks = []
        if depth == 1:
            blocks.append(
                DepthwiseSwiGLUBlock3D(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    zero_init_out=zero_init_out,
                )
            )
        else:
            for _ in range(depth - 1):
                blocks.append(
                    DepthwiseSwiGLUBlock3D(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dropout=dropout,
                        zero_init_out=False,
                    )
                )
            blocks.append(
                DepthwiseSwiGLUBlock3D(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    zero_init_out=zero_init_out,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class DepthwiseSwiGLUBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        dropout: float = 0.0,
        zero_init_out: bool = False,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pw_in = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pw_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        if zero_init_out:
            nn.init.zeros_(self.pw_out.weight)
            nn.init.zeros_(self.pw_out.bias)

    def forward(self, x):
        w = self.dw.weight
        if x.dtype != w.dtype or x.device != w.device:
            x = x.to(device=w.device, dtype=w.dtype)
        x = self.dw(x)
        x = self.pw_in(x)
        u, v = torch.chunk(x, 2, dim=1)
        x = u * F.silu(v)
        x = self.dropout(x)
        x = self.pw_out(x)
        return x


class DepthwiseSwiGLUStack2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        depth: int = 1,
        dropout: float = 0.0,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >=1, got {depth}")
        blocks = []
        if depth == 1:
            blocks.append(
                DepthwiseSwiGLUBlock2D(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    zero_init_out=zero_init_out,
                )
            )
        else:
            for _ in range(depth - 1):
                blocks.append(
                    DepthwiseSwiGLUBlock2D(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dropout=dropout,
                        zero_init_out=False,
                    )
                )
            blocks.append(
                DepthwiseSwiGLUBlock2D(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    zero_init_out=zero_init_out,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)
