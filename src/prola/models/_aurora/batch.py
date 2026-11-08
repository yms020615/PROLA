import dataclasses
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as RGI

from prola.models._aurora.normalisation import (
    normalise_atmos_var,
    normalise_surf_var,
    unnormalise_atmos_var,
    unnormalise_surf_var,
)

__all__ = ["Metadata", "Batch"]


@dataclasses.dataclass
class Metadata:

    lat: torch.Tensor
    lon: torch.Tensor
    time: tuple[datetime, ...]
    atmos_levels: tuple[int | float, ...]
    rollout_step: int = 0

    def __post_init__(self):
        if not (torch.all(self.lat <= 90) and torch.all(self.lat >= -90)):
            raise ValueError("Latitudes must be in the range [-90, 90].")
        if not (torch.all(self.lon >= 0) and torch.all(self.lon < 360)):
            raise ValueError("Longitudes must be in the range [0, 360).")

        if self.lat.dim() == self.lon.dim() == 1:
            if not torch.all(self.lat[1:] - self.lat[:-1] < 0):
                raise ValueError("Latitudes must be strictly decreasing.")
            if not torch.all(self.lon[1:] - self.lon[:-1] > 0):
                raise ValueError("Longitudes must be strictly increasing.")

        elif self.lat.dim() == self.lon.dim() == 2:
            if not torch.all(self.lat[1:, :] - self.lat[:-1, :]):
                raise ValueError("Latitudes must be strictly decreasing along every column.")
            if not torch.all(self.lon[:, 1:] - self.lon[:, :-1] > 0):
                raise ValueError("Longitudes must be strictly increasing along every row.")

        else:
            raise ValueError(
                "The latitudes and longitudes must either both be vectors or both be matrices."
            )


@dataclasses.dataclass
class Batch:

    surf_vars: dict[str, torch.Tensor]
    static_vars: dict[str, torch.Tensor]
    atmos_vars: dict[str, torch.Tensor]
    metadata: Metadata

    @property
    def spatial_shape(self) -> tuple[int, int]:

        return next(iter(self.surf_vars.values())).shape[-2:]

    def normalise(self, surf_stats: dict[str, tuple[float, float]]) -> "Batch":

        return Batch(
            surf_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: normalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def unnormalise(self, surf_stats: dict[str, tuple[float, float]]) -> "Batch":

        return Batch(
            surf_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: unnormalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def crop(self, patch_size: int) -> "Batch":

        h, w = self.spatial_shape

        if w % patch_size != 0:
            raise ValueError("Width of the data must be a multiple of the patch size.")

        if h % patch_size == 0:
            return self
        elif h % patch_size == 1:
            return Batch(
                surf_vars={k: v[..., :-1, :] for k, v in self.surf_vars.items()},
                static_vars={k: v[..., :-1, :] for k, v in self.static_vars.items()},
                atmos_vars={k: v[..., :-1, :] for k, v in self.atmos_vars.items()},
                metadata=Metadata(
                    lat=self.metadata.lat[:-1],
                    lon=self.metadata.lon,
                    atmos_levels=self.metadata.atmos_levels,
                    time=self.metadata.time,
                    rollout_step=self.metadata.rollout_step,
                ),
            )
        else:
            raise ValueError(
                f"There can at most be one latitude too many, "
                f"but there are {h % patch_size} too many."
            )

    def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        return Batch(
            surf_vars={k: f(v) for k, v in self.surf_vars.items()},
            static_vars={k: f(v) for k, v in self.static_vars.items()},
            atmos_vars={k: f(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=f(self.metadata.lat),
                lon=f(self.metadata.lon),
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to(self, device: str | torch.device) -> "Batch":

        return self._fmap(lambda x: x.to(device))

    def type(self, t: type) -> "Batch":

        lat = self.metadata.lat
        lon = self.metadata.lon
        if lat.dtype not in (torch.float32, torch.float64):
            lat = lat.float()
        if lon.dtype not in (torch.float32, torch.float64):
            lon = lon.float()

        return Batch(
            surf_vars={k: v.type(t) for k, v in self.surf_vars.items()},
            static_vars={k: v.type(t) for k, v in self.static_vars.items()},
            atmos_vars={k: v.type(t) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=lat,
                lon=lon,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def regrid(self, res: float) -> "Batch":

        shape = (round(180 / res) + 1, round(360 / res))
        lat_new = torch.from_numpy(np.linspace(90, -90, shape[0]))
        lon_new = torch.from_numpy(np.linspace(0, 360, shape[1], endpoint=False))
        interpolate_res = partial(
            interpolate,
            lat=self.metadata.lat,
            lon=self.metadata.lon,
            lat_new=lat_new,
            lon_new=lon_new,
        )

        return Batch(
            surf_vars={k: interpolate_res(v) for k, v in self.surf_vars.items()},
            static_vars={k: interpolate_res(v) for k, v in self.static_vars.items()},
            atmos_vars={k: interpolate_res(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=lat_new,
                lon=lon_new,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to_netcdf(self, path: str | Path) -> None:

        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.Dataset(
            {
                **{
                    f"surf_{k}": (("batch", "history", "latitude", "longitude"), _np(v))
                    for k, v in self.surf_vars.items()
                },
                **{
                    f"static_{k}": (("latitude", "longitude"), _np(v))
                    for k, v in self.static_vars.items()
                },
                **{
                    f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), _np(v))
                    for k, v in self.atmos_vars.items()
                },
            },
            coords={
                "latitude": _np(self.metadata.lat),
                "longitude": _np(self.metadata.lon),
                "time": list(self.metadata.time),
                "level": list(self.metadata.atmos_levels),
                "rollout_step": self.metadata.rollout_step,
            },
        )
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "Batch":

        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.load_dataset(path, engine="netcdf4")

        surf_vars: List[str] = []
        static_vars: List[str] = []
        atmos_vars: List[str] = []

        for k in ds:
            if k.startswith("surf_"):
                surf_vars.append(k.removeprefix("surf_"))
            elif k.startswith("static_"):
                static_vars.append(k.removeprefix("static_"))
            elif k.startswith("atmos_"):
                atmos_vars.append(k.removeprefix("atmos_"))

        return Batch(
            surf_vars={k: torch.from_numpy(ds[f"surf_{k}"].values) for k in surf_vars},
            static_vars={k: torch.from_numpy(ds[f"static_{k}"].values) for k in static_vars},
            atmos_vars={k: torch.from_numpy(ds[f"atmos_{k}"].values) for k in atmos_vars},
            metadata=Metadata(
                lat=torch.from_numpy(ds.latitude.values),
                lon=torch.from_numpy(ds.longitude.values),
                time=tuple(ds.time.values.astype("datetime64[s]").tolist()),
                atmos_levels=tuple(ds.level.values),
                rollout_step=int(ds.rollout_step.values),
            ),
        )


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def interpolate(
    v: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_new: torch.Tensor,
    lon_new: torch.Tensor,
) -> torch.Tensor:

    return torch.from_numpy(
        interpolate_numpy(
            v.double().numpy(),
            lat.double().numpy(),
            lon.double().numpy(),
            lat_new.double().numpy(),
            lon_new.double().numpy(),
        )
    ).float()


def interpolate_numpy(
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_new: np.ndarray,
    lon_new: np.ndarray,
) -> np.ndarray:

    assert (np.diff(lon) > 0).all()
    lon = np.concatenate((lon[-1:] - 360, lon, lon[:1] + 360))

    batch_shape = v.shape[:-2]
    v = v.reshape(-1, *v.shape[-2:])

    vs_regridded = []
    for vi in v:

        vi = np.concatenate((vi[:, -1:], vi, vi[:, :1]), axis=1)

        rgi = RGI(
            (lat, lon),
            vi,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        lat_new_grid, lon_new_grid = np.meshgrid(
            lat_new,
            lon_new,
            indexing="ij",
            sparse=True,
        )
        vs_regridded.append(rgi((lat_new_grid, lon_new_grid)))

    v_regridded = np.stack(vs_regridded, axis=0)
    v_regridded = v_regridded.reshape(*batch_shape, lat_new.shape[0], lon_new.shape[0])

    return v_regridded
