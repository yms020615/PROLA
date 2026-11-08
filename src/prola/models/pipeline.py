from datetime import datetime, timezone

import torch
from torch import nn
from torch.nn import functional as F

from ._aurora import Batch, Metadata
from ._aurora.normalisation import (
    normalise_surf_var,
    normalise_atmos_var,
    unnormalise_surf_var,
    unnormalise_atmos_var,
)
from .heads import SpatialCorrectionHead, TemporalCorrectionHead


def normalize_surface(value, variables):
    return torch.stack(
        [normalise_surf_var(value[:, :, i], var, None) for i, var in enumerate(variables)], dim=2
    )


def normalize_upper(value, variables, levels):
    return torch.stack(
        [normalise_atmos_var(value[:, :, i], var, levels) for i, var in enumerate(variables)], dim=2
    )


def resize(value, shape):
    leading = value.shape[:-2]
    return F.interpolate(
        value.reshape(-1, 1, *value.shape[-2:]), size=shape, mode="bilinear", align_corners=False
    ).reshape(*leading, *shape)


class WeatherModel(nn.Module):
    def __init__(self, backbone, task, grid):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.register_buffer(
            "latitude", torch.as_tensor(grid["latitude"], dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "longitude", torch.as_tensor(grid["longitude"], dtype=torch.float32), persistent=False
        )
        for key in ("lsm", "z", "slt"):
            self.register_buffer(
                "static_" + key, torch.as_tensor(grid[key], dtype=torch.float32), persistent=False
            )
        if task["name"] != "precipitation":
            channels = len(task["upper"]) * len(task["levels"])
            self.spatial_corrector = SpatialCorrectionHead(len(task["surface"]), channels)
            self.temporal_corrector = TemporalCorrectionHead(
                len(task["surface"]), channels, num_leads=len(task["leads"])
            )

    def forward(self, sample):
        shape = (len(self.latitude), len(self.longitude))
        surface = resize(sample["input_surface"], shape)
        upper = resize(sample["input_upper"], shape)
        times = tuple(
            datetime.fromtimestamp(int(t), timezone.utc).replace(tzinfo=None)
            for t in sample["anchor_time"].detach().cpu()
        )
        batch = Batch(
            surf_vars={v: surface[:, :, i] for i, v in enumerate(self.task["surface"])},
            atmos_vars={v: upper[:, :, i] for i, v in enumerate(self.task["upper"])},
            static_vars={v: getattr(self, "static_" + v) for v in ("lsm", "z", "slt")},
            metadata=Metadata(
                lat=self.latitude,
                lon=self.longitude,
                time=times,
                atmos_levels=tuple(self.task["levels"]),
            ),
        )
        endpoint_s, endpoint_u = self.backbone(batch)
        if self.task["name"] == "precipitation":
            return {"precipitation": endpoint_s}
        surface = normalize_surface(surface, self.task["surface"])
        upper = normalize_upper(upper, self.task["upper"], tuple(self.task["levels"]))
        ds, du = self.spatial_corrector(surface, upper, endpoint_s, endpoint_u)
        endpoint_s, endpoint_u = endpoint_s + ds, endpoint_u + du
        fractions = surface.new_tensor(self.task["leads"]) / max(self.task["leads"])
        current_s, current_u = surface[:, -1], upper[:, -1]
        scaffold_s = (
            current_s[:, None] + fractions.view(1, -1, 1, 1, 1) * (endpoint_s - current_s)[:, None]
        )
        scaffold_u = (
            current_u[:, None]
            + fractions.view(1, -1, 1, 1, 1, 1) * (endpoint_u - current_u)[:, None]
        )
        ds, du = self.temporal_corrector(
            current_s, current_u, endpoint_s, endpoint_u, scaffold_s, scaffold_u
        )
        return {"surface": scaffold_s + ds, "upper": scaffold_u + du}

    def denormalize(self, output):
        if self.task["name"] == "precipitation":
            return {
                "precipitation": unnormalise_surf_var(
                    output["precipitation"], "tp", None
                ).clamp_min(0)
            }
        surface = torch.stack(
            [
                unnormalise_surf_var(output["surface"][:, :, i], v, None)
                for i, v in enumerate(self.task["surface"])
            ],
            dim=2,
        )
        upper = torch.stack(
            [
                unnormalise_atmos_var(output["upper"][:, :, i], v, tuple(self.task["levels"]))
                for i, v in enumerate(self.task["upper"])
            ],
            dim=2,
        )
        return {"surface": surface, "upper": upper}
