from datetime import datetime, timezone

import numpy as np
from torch.utils.data import Dataset

from .common import anchors


class PrecipitationDataset(Dataset):
    def __init__(self, root, config, split, task):
        self.root = root / config["inputs"]
        self.task = task
        self.anchors = anchors(*config[split], task["input_lag"], task["leads"], 12)
        with np.load(self.root / "metadata.npz", allow_pickle=False) as meta:
            names = dict(
                latitude="lat", longitude="lon", lsm="static_lsm", z="static_z", slt="static_slt"
            )
            self.grid = {k: np.asarray(meta[v], dtype="float32") for k, v in names.items()}
            for key, expected in (
                ("surf_vars", task["surface"]),
                ("upper_vars", task["upper"]),
                ("levels", task["levels"]),
            ):
                if key in meta and tuple(meta[key]) != tuple(expected):
                    raise ValueError(f"Unexpected {key} in precipitation metadata.")
        if not len(self.anchors):
            raise ValueError("No complete forecast windows exist in this split.")

    def frame(self, stamp):
        name = datetime.fromtimestamp(int(stamp), timezone.utc).strftime("%Y%m%d%H")
        array = np.load(self.root / f"{name}.npy", allow_pickle=False)
        expected = (5, 14, len(self.grid["latitude"]), len(self.grid["longitude"]))
        if array.shape != expected:
            raise ValueError(f"Expected precipitation frame {expected}, received {array.shape}.")
        return array.astype("float32")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        t = int(self.anchors[index])
        inputs = np.stack([self.frame(t - 43200), self.frame(t)])
        targets = np.stack([self.frame(t + h * 3600)[4, 0] for h in self.task["leads"]])
        return dict(
            input_surface=inputs[:, :, 0],
            input_upper=inputs[:, :, 1:],
            target_precipitation=targets,
            anchor_time=np.int64(t),
        )
