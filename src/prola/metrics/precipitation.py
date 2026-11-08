from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from .weather import latitude_weights, spatial_acc


class PrecipitationClimatology:
    def __init__(self, path, grid):
        variable = "total_precipitation_6hr"
        names = dict(
            mean=variable,
            threshold=variable + "_seeps_threshold",
            dry=variable + "_seeps_dry_fraction",
        )
        self.fields = {}
        for key, name in names.items():
            with xr.open_dataset(path / f"{name}.nc") as ds:
                da = ds[name].sel(latitude=grid["latitude"], longitude=grid["longitude"])
                if key == "dry":
                    da = da.mean(("hour", "dayofyear"))
                self.fields[key] = da.load()

    def select(self, timestamps):
        stamps = np.asarray(timestamps)
        times = [datetime.fromtimestamp(int(t), timezone.utc) for t in stamps.ravel()]
        day = xr.DataArray([t.timetuple().tm_yday for t in times], dims="sample")
        hour = xr.DataArray([t.hour for t in times], dims="sample")
        result = {}
        for key in ("mean", "threshold"):
            array = (
                self.fields[key]
                .sel(dayofyear=day, hour=hour)
                .transpose("sample", "latitude", "longitude")
                .values
            )
            result[key] = array.reshape(*stamps.shape, *array.shape[-2:])
        result["dry"] = self.fields["dry"].transpose("latitude", "longitude").values
        return result


class PrecipitationMetrics:
    def __init__(self, task, grid, climatology):
        self.task, self.latitude, self.climatology = task, grid["latitude"], climatology
        self.totals = np.zeros((4, len(task["leads"])), dtype="float64")
        self.count = 0

    def update(self, prediction, sample):
        pred = prediction["precipitation"].double().clamp_min(0)
        target = sample["target_precipitation"].double()
        times = (
            sample["anchor_time"].cpu().numpy()[:, None]
            + np.asarray(self.task["leads"])[None, :] * 3600
        )
        climate = {
            k: torch.as_tensor(v, device=pred.device, dtype=pred.dtype)
            for k, v in self.climatology.select(times).items()
        }
        weight = latitude_weights(self.latitude, pred)
        self.totals[0] += ((pred - target).square() * weight).mean((-2, -1)).sum(0).cpu().numpy()
        self.totals[1] += spatial_acc(pred, target, climate["mean"], weight).sum(0).cpu().numpy()
        p1 = climate["dry"]
        valid = (p1 > 0.1) & (p1 < 0.85) & torch.isfinite(p1)
        p1 = p1.nan_to_num(0.5).clamp(1e-6, 1 - 1e-6)
        zero = torch.zeros_like(p1)
        matrix = 0.5 * torch.stack(
            (
                zero,
                1 / (1 - p1),
                4 / (1 - p1),
                1 / p1,
                zero,
                3 / (1 - p1),
                1 / p1 + 3 / (2 + p1),
                3 / (2 + p1),
                zero,
            ),
            dim=-1,
        )
        p_cat = torch.where(pred < 1e-4, 0, torch.where(pred >= climate["threshold"], 2, 1))
        t_cat = torch.where(target < 1e-4, 0, torch.where(target >= climate["threshold"], 2, 1))
        matrix = matrix.expand(*pred.shape, 9)
        score = matrix.gather(-1, (3 * p_cat + t_cat).unsqueeze(-1)).squeeze(-1)
        valid = (
            valid
            & torch.isfinite(pred)
            & torch.isfinite(target)
            & torch.isfinite(climate["threshold"])
        )
        valid_weight = weight * valid
        self.totals[2] += (torch.where(valid, score, 0) * weight).sum((0, 2, 3)).cpu().numpy()
        self.totals[3] += valid_weight.sum((0, 2, 3)).cpu().numpy()
        self.count += len(pred)

    def result(self):
        if not self.count or np.any(self.totals[3] == 0):
            raise ValueError("Evaluation contains no samples or no valid SEEPS locations.")
        result = {}
        for i, lead in enumerate(self.task["leads"]):
            result[f"tp/rmse_cm/{lead}h"] = float(np.sqrt(self.totals[0, i] / self.count) * 100)
            result[f"tp/acc/{lead}h"] = float(self.totals[1, i] / self.count)
            result[f"tp/seeps/{lead}h"] = float(self.totals[2, i] / self.totals[3, i])
        return result
