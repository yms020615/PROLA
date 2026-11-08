import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


SURFACE = (
    ("2m_temperature", "t2m"),
    ("10m_u_component_of_wind", "u10"),
    ("10m_v_component_of_wind", "v10"),
)
UPPER = (
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("specific_humidity", "q"),
    ("geopotential", "z"),
)


def timestamp(value):
    return int(np.datetime64(str(value), "s").astype("int64"))


def anchors(start, end, lag, leads, stride):
    start, end = timestamp(start), timestamp(end)
    return np.arange(start + lag * 3600, end - max(leads) * 3600, stride * 3600, dtype="int64")


class ERA5Store:
    def __init__(self, root, degree):
        self.root, self.degree = Path(root), str(degree)
        self._files = OrderedDict()
        self._pid = os.getpid()

    def array(self, stamp, variable, levels=None):
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        when = np.datetime64(int(stamp), "s")
        prefix, short = variable
        year = str(when)[:4]
        key = (prefix, year)
        if key not in self._files:
            if len(self._files) >= 16:
                self._files.popitem(last=False)[1].close()
            self._files[key] = xr.open_dataset(
                self.root / prefix / f"{prefix}_{year}_{self.degree}deg.nc"
            )
        self._files.move_to_end(key)
        da = self._files[key][short].sel(time=when)
        renames = {
            k: v
            for k, v in (
                ("lat", "latitude"),
                ("lon", "longitude"),
                ("pressure_level", "level"),
                ("plev", "level"),
            )
            if k in da.dims
        }
        da = da.rename(renames).sortby("latitude", ascending=False).sortby("longitude")
        if levels is not None:
            coordinate = da.level.values
            selection = np.asarray(levels) * (100 if coordinate.max() > 2000 else 1)
            da = da.sel(level=selection).transpose("level", "latitude", "longitude")
        else:
            da = da.transpose("latitude", "longitude")
        return da

    def frame(self, stamp, levels):
        surface = np.stack([self.array(stamp, v).values for v in SURFACE]).astype("float32")
        upper = np.stack([self.array(stamp, v, levels).values for v in UPPER]).astype("float32")
        return surface, upper

    def close(self):
        for ds in self._files.values():
            ds.close()
        self._files.clear()

    def __getstate__(self):
        return dict(self.__dict__, _files=OrderedDict())


def static_grid(path, latitude, longitude):
    latitude, longitude = np.asarray(latitude, dtype="float32"), np.asarray(
        longitude, dtype="float32"
    )
    grid = dict(latitude=latitude, longitude=longitude)
    target = np.stack(np.meshgrid(latitude, longitude, indexing="ij"), axis=-1)
    with xr.open_dataset(path) as ds:
        for key in ("lsm", "z", "slt"):
            da = ds[key].squeeze(drop=True).sortby("latitude").sortby("longitude")
            lon = da.longitude.values
            values = da.transpose("latitude", "longitude").values
            points = target.copy()
            if lon.min() < 0:
                points[..., 1] = (points[..., 1] + 180) % 360 - 180
            else:
                points[..., 1] %= 360
            interpolator = RegularGridInterpolator(
                (da.latitude.values, lon), values, bounds_error=False, fill_value=None
            )
            grid[key] = interpolator(points).astype("float32")
    return grid
