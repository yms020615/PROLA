import torch
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from prola.models._aurora.model.fourier import FourierExpansion

__all__ = ["pos_scale_enc"]


def patch_root_area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:

    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = (
        6371**2
        * torch.pi
        * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min)))
        * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))
    )

    assert (area > 0.0).all()
    return torch.sqrt(area)


def pos_scale_enc_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)

    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)

    return pos_encode, scale_encode


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:

    assert lat.dim() == 1
    assert lon.dim() == 1

    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)

    return grid


def pos_scale_enc(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> torch.Tensor:

    if lat.dim() == lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, "
            f"but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )

    grid = grid[None]

    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )

    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)
