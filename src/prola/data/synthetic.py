import numpy as np
import torch
from torch.utils.data import Dataset

from prola.models._aurora.normalisation import locations, scales


class SyntheticDataset(Dataset):
    def __init__(self, task, count=4):
        self.task, self.count = task, count
        self.grid = dict(
            latitude=np.linspace(60, 0, 8, dtype="float32"),
            longitude=np.linspace(75, 120, 16, dtype="float32"),
        )
        self.grid.update({k: np.zeros((8, 16), dtype="float32") for k in ("lsm", "z", "slt")})

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = torch.Generator().manual_seed(100 + index)
        steps = 2 + len(self.task["leads"])
        surface = torch.randn(steps, len(self.task["surface"]), 8, 16, generator=rng) * 0.1
        upper = (
            torch.randn(
                steps, len(self.task["upper"]), len(self.task["levels"]), 8, 16, generator=rng
            )
            * 0.1
        )
        for i, name in enumerate(self.task["surface"]):
            surface[:, i] = surface[:, i] * scales[name] + locations[name]
            if name == "tp":
                surface[:, i].clamp_min_(0)
        for i, name in enumerate(self.task["upper"]):
            for j, level in enumerate(self.task["levels"]):
                key = f"{name}_{level}"
                upper[:, i, j] = upper[:, i, j] * scales[key] + locations[key]
        result = dict(
            input_surface=surface[:2],
            input_upper=upper[:2],
            anchor_time=np.int64(1514808000 + index * self.task["input_lag"] * 3600),
        )
        if self.task["name"] == "precipitation":
            result["target_precipitation"] = surface[2:, 4]
        else:
            result.update(target_surface=surface[2:], target_upper=upper[2:])
        return result


class SyntheticClimatology:
    def __init__(self, grid):
        self.shape = (len(grid["latitude"]), len(grid["longitude"]))

    def select(self, times):
        return dict(
            mean=np.zeros((*times.shape, *self.shape)),
            threshold=np.full((*times.shape, *self.shape), 0.001),
            dry=np.full(self.shape, 0.5),
        )
