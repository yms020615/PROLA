import json
import os
from collections import OrderedDict

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset

from .common import ERA5Store, SURFACE, UPPER, timestamp, static_grid


CHANNELS = [("2t", 0), ("10si", 0), ("10wdir", 0)] + [
    (v, p) for v in ("z", "t", "u", "v", "r") for p in (500, 850)
]


def wind_basis(latitude, longitude):
    phi, lam = np.deg2rad(latitude.astype("float64")), np.deg2rad(longitude.astype("float64"))
    xyz = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    tangent = np.gradient(xyz, axis=1, edge_order=2)
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.stack((-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)), axis=-1)
    e, n = (tangent * east).sum(-1), (tangent * north).sum(-1)
    length = np.hypot(e, n)
    if not np.all(length > 0):
        raise ValueError("Degenerate CERRA coordinate grid.")
    return (e / length).astype("float32"), (n / length).astype("float32")


def specific_humidity(rh, temperature, pressure):
    celsius = temperature - 273.15
    vapour = np.clip(rh, 0, 100) * 0.01 * 611.2 * np.exp(17.67 * celsius / (celsius + 243.5))
    return 0.622 * vapour / (pressure - 0.378 * vapour)


class CERRADataset(Dataset):
    def __init__(self, root, config, split, task):
        self.root = root / config["targets"]
        self.inputs = ERA5Store(root / config["inputs"], "5.625")
        self.task = task
        self.locations, self.metadata = {}, {}
        self.handles = OrderedDict()
        self.pid = os.getpid()
        start, end = map(timestamp, config[split])
        first = True
        for path in sorted(self.root.glob("cerra_central_europe_*.h5")):
            month = path.stem.rsplit("_", 1)[1]
            date = timestamp(f"{month[:4]}-{month[4:]}-01")
            next_date = int(
                (np.datetime64(f"{month[:4]}-{month[4:]}", "M") + 1)
                .astype("datetime64[s]")
                .astype("int64")
            )
            if next_date <= start or date >= end:
                continue
            with h5py.File(path, "r") as f:
                channels = json.loads(f.attrs["channels"])
                if channels != [list(x) for x in CHANNELS]:
                    raise ValueError(f"Unexpected channel order in {path.name}.")
                lat, lon = f["latitude"][:], f["longitude"][:]
                if first:
                    self.latitude, self.longitude = lat, lon
                    first = False
                elif not (
                    np.array_equal(lat, self.latitude) and np.array_equal(lon, self.longitude)
                ):
                    raise ValueError("CERRA shards use different coordinate grids.")
                self.metadata[month] = json.loads(f.attrs["variable_metadata"])
                if self.metadata[month]["2"]["units"] != "Degree true":
                    raise ValueError("Surface wind direction must be relative to true north.")
                for i, t in enumerate(f["time"][:]):
                    if start <= t < end:
                        if int(t) in self.locations:
                            raise ValueError("Duplicate CERRA timestamps.")
                        self.locations[int(t)] = (month, i)
        self.anchors = np.asarray(
            [
                t
                for t in sorted(self.locations)
                if t % 21600 == 0 and all(t + h * 3600 in self.locations for h in (-6, 3, 6))
            ],
            dtype="int64",
        )
        if not len(self.anchors):
            raise ValueError("No complete CERRA forecast windows exist in this split.")
        self.east, self.north = wind_basis(self.latitude, self.longitude)
        height, width = self.latitude.shape
        lat = self.latitude[:, width // 2]
        lon = self.longitude[height // 2] % 360
        self.flip = lat[0] < lat[-1]
        self.order = np.argsort(lon)
        self.grid = static_grid(
            root / config["static"], lat[::-1] if self.flip else lat, lon[self.order]
        )
        self.points = np.stack(
            (self.latitude.ravel(), ((self.longitude + 180) % 360 - 180).ravel()), axis=-1
        )

    def orient(self, array):
        if self.flip:
            array = array[..., ::-1, :]
        return np.ascontiguousarray(array[..., self.order], dtype="float32")

    def input(self, stamp):
        fields = []
        for variable in SURFACE + UPPER:
            da = self.inputs.array(
                stamp, variable, self.task["levels"] if variable in UPPER else None
            )
            lat, lon = da.latitude.values, (da.longitude.values + 180) % 360 - 180
            iy, ix = np.argsort(lat), np.argsort(lon)
            values = da.values[..., iy, :][..., ix]
            upper = values.ndim == 3
            interpolator = RegularGridInterpolator(
                (lat[iy], lon[ix]), np.moveaxis(values, 0, -1) if upper else values
            )
            result = interpolator(self.points)
            fields.append(
                result.T.reshape(2, *self.latitude.shape)
                if upper
                else result.reshape(self.latitude.shape)
            )
        return np.stack(fields[:3]), np.stack(fields[3:])

    def target(self, stamp):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        month, index = self.locations[int(stamp)]
        if month not in self.handles:
            if len(self.handles) >= 2:
                self.handles.popitem(last=False)[1].close()
            self.handles[month] = h5py.File(self.root / f"cerra_central_europe_{month}.h5", "r")
        self.handles.move_to_end(month)
        values = self.handles[month]["fields"][index]
        angle = np.deg2rad(values[2])
        surface = np.stack((values[0], -values[1] * np.sin(angle), -values[1] * np.cos(angle)))
        u, v = values[7:9].copy(), values[9:11].copy()
        for j, level in enumerate(self.task["levels"]):
            flags = [self.metadata[month][str(k + j)]["uvRelativeToGrid"] for k in (7, 9)]
            if flags[0] != flags[1] or flags[0] not in (0, 1):
                raise ValueError("Inconsistent CERRA wind orientation.")
            if flags[0]:
                u[j] = values[7 + j] * self.east - values[9 + j] * self.north
                v[j] = values[7 + j] * self.north + values[9 + j] * self.east
        q = np.stack(
            [
                specific_humidity(values[11 + j], values[5 + j], level * 100)
                for j, level in enumerate(self.task["levels"])
            ]
        )
        return surface, np.stack((values[5:7], u, v, q, values[3:5]))

    def __getitem__(self, index):
        t = int(self.anchors[index])
        inputs = [self.input(t - 21600), self.input(t)]
        targets = [self.target(t + h * 3600) for h in self.task["leads"]]
        return dict(
            input_surface=self.orient(np.stack([x[0] for x in inputs])),
            input_upper=self.orient(np.stack([x[1] for x in inputs])),
            target_surface=self.orient(np.stack([y[0] for y in targets])),
            target_upper=self.orient(np.stack([y[1] for y in targets])),
            anchor_time=np.int64(t),
        )

    def __len__(self):
        return len(self.anchors)

    def close(self):
        for file in self.handles.values():
            file.close()
        self.handles.clear()

    def __getstate__(self):
        return dict(self.__dict__, handles=OrderedDict())
