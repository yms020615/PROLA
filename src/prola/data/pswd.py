import numpy as np
from torch.utils.data import Dataset

from .common import ERA5Store, SURFACE, anchors, static_grid


class PSWDDataset(Dataset):
    def __init__(self, root, config, split, task):
        self.task = task
        self.inputs = ERA5Store(root / str(config["inputs"]), "5.625")
        self.targets = ERA5Store(root / config["targets"], "1.5")
        self.anchors = anchors(*config[split], task["input_lag"], task["leads"], 6)
        if not len(self.anchors):
            raise ValueError("No complete forecast windows exist in this split.")
        da = self.targets.array(self.anchors[0] + 3600, SURFACE[0])
        latitude, longitude = da.latitude.values, da.longitude.values
        self.height = len(latitude) - len(latitude) % 4
        self.width = len(longitude) - len(longitude) % 4
        self.grid = static_grid(
            root / config["static"], latitude[: self.height], longitude[: self.width]
        )

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        t = int(self.anchors[index])
        levels = self.task["levels"]
        inputs = [self.inputs.frame(t + h * 3600, levels) for h in (-6, 0)]
        targets = [self.targets.frame(t + h * 3600, levels) for h in self.task["leads"]]
        return dict(
            input_surface=np.stack([x[0] for x in inputs]),
            input_upper=np.stack([x[1] for x in inputs]),
            target_surface=np.stack([y[0] for y in targets])[..., : self.height, : self.width],
            target_upper=np.stack([y[1] for y in targets])[..., : self.height, : self.width],
            anchor_time=np.int64(t),
        )
