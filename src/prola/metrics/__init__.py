from pathlib import Path

import numpy as np


def build_metrics(config, task, grid, root, synthetic=False):
    if task["name"] == "precipitation":
        from .precipitation import PrecipitationClimatology, PrecipitationMetrics

        if synthetic:
            from prola.data.synthetic import SyntheticClimatology

            climate = SyntheticClimatology(grid)
        else:
            climate = PrecipitationClimatology(Path(root) / config["data"]["climatology"], grid)
        return PrecipitationMetrics(task, grid, climate)
    from .weather import WeatherMetrics, VARIABLES

    climate = None
    if task["name"] == "pswd":
        if synthetic:
            climate = {
                v: np.zeros((len(grid["latitude"]), len(grid["longitude"])), dtype="float32")
                for v in VARIABLES
            }
        else:
            with np.load(Path(root) / config["data"]["climatology"], allow_pickle=False) as f:
                climate = {v: f[v] for v in VARIABLES}
    return WeatherMetrics(task, grid, climate)
