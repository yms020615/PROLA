import torch

__all__ = ["area", "compute_patch_areas", "radius_earth"]

radius_earth = 6378137 / 1000


def area(polygon: torch.Tensor) -> torch.Tensor:

    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]

    rad = torch.deg2rad

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))

    area = area * radius_earth * radius_earth / 2

    return torch.abs(area)


def expand_matrix(matrix: torch.Tensor) -> torch.Tensor:

    matrix = torch.cat(
        (
            2 * matrix[0:1] - matrix[1:2],
            matrix,
            2 * matrix[-1:] - matrix[-2:-1],
        ),
        dim=0,
    )

    matrix = torch.cat(
        (
            2 * matrix[:, 0:1] - matrix[:, 1:2],
            matrix,
            2 * matrix[:, -1:] - matrix[:, -2:-1],
        ),
        dim=1,
    )

    return matrix


def compute_patch_areas(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:

    if not (lat.dim() == lon.dim() == 2):
        raise ValueError("`lat` and `lon` must both be matrices.")
    if lat.shape != lat.shape:
        raise ValueError("`lat` and `lon` must have the same shape.")

    if not torch.all(lat[1:] - lat[:-1] <= 0):
        raise ValueError("`lat` must be decreasing along rows.")

    if not torch.all(lon[:, 1:] - lon[:, :-1] >= 0):
        raise ValueError("`lon` must be increasing along columns.")

    lat = expand_matrix(lat)
    lon = expand_matrix(lon)

    lat = torch.clamp(lat, -90, 90)

    lat_midpoints = (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:]) / 4
    lon_midpoints = (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:]) / 4

    top_left = torch.stack((lat_midpoints[1:, :-1], lon_midpoints[1:, :-1]), dim=-1)
    top_right = torch.stack((lat_midpoints[1:, 1:], lon_midpoints[1:, 1:]), dim=-1)
    bottom_left = torch.stack((lat_midpoints[:-1, :-1], lon_midpoints[:-1, :-1]), dim=-1)
    bottom_right = torch.stack((lat_midpoints[:-1, 1:], lon_midpoints[:-1, 1:]), dim=-1)
    polygon = torch.stack((top_left, top_right, bottom_right, bottom_left), dim=-2)

    return area(polygon)
