import numpy as np
import torch


VARIABLES = ("z500", "t850", "2t", "10u", "10v")


def reported_fields(output, task):
    fields = {name: output["surface"][:, :, i] for i, name in enumerate(task["surface"])}
    for name, variable, level in (("z500", "z", 500), ("t850", "t", 850)):
        fields[name] = output["upper"][
            :, :, task["upper"].index(variable), task["levels"].index(level)
        ]
    return {key: fields[key] for key in VARIABLES}


def latitude_weights(latitude, reference):
    weights = (
        torch.as_tensor(latitude, device=reference.device, dtype=reference.dtype).deg2rad().cos()
    )
    return (weights / weights.mean()).view(1, 1, -1, 1)


def spatial_acc(prediction, target, climatology, weights):
    p, t = prediction - climatology, target - climatology
    numerator = (p * t * weights).sum((-2, -1))
    denominator = (
        (p.square() * weights).sum((-2, -1)) * (t.square() * weights).sum((-2, -1))
    ).sqrt()
    return numerator / denominator.clamp_min(1e-12)


class WeatherMetrics:
    def __init__(self, task, grid, climatology=None):
        self.task, self.latitude = task, grid["latitude"]
        self.climatology = climatology
        self.totals = np.zeros((2, len(VARIABLES), len(task["leads"])), dtype="float64")
        self.count = 0

    def update(self, prediction, sample):
        fields = reported_fields(prediction, self.task)
        targets = reported_fields(
            dict(surface=sample["target_surface"], upper=sample["target_upper"]), self.task
        )
        for i, name in enumerate(VARIABLES):
            pred, target = fields[name].double(), targets[name].double()
            weights = latitude_weights(self.latitude, pred)
            rmse = ((pred - target).square() * weights).mean((-2, -1)).sqrt()
            self.totals[0, i] += rmse.sum(0).cpu().numpy()
            if self.climatology is not None:
                climate = torch.as_tensor(
                    self.climatology[name], dtype=pred.dtype, device=pred.device
                )
                if climate.shape != pred.shape[-2:]:
                    raise ValueError(f"Climatology grid mismatch for {name}.")
                self.totals[1, i] += (
                    spatial_acc(pred, target, climate, weights).sum(0).cpu().numpy()
                )
        self.count += len(sample["anchor_time"])

    def result(self):
        if not self.count:
            raise ValueError("No evaluation samples were accumulated.")
        result = {
            f"{name}/rmse/{h}h": float(self.totals[0, i, j] / self.count)
            for i, name in enumerate(VARIABLES)
            for j, h in enumerate(self.task["leads"])
        }
        if self.climatology is not None:
            result.update(
                {
                    f"{name}/acc/{h}h": float(self.totals[1, i, j] / self.count)
                    for i, name in enumerate(VARIABLES)
                    for j, h in enumerate(self.task["leads"])
                }
            )
        return result
