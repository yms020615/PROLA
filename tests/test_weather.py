import importlib.util
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


WEATHER_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("xarray", "scipy", "einops", "timm", "huggingface_hub", "h5py")
)


@unittest.skipUnless(WEATHER_AVAILABLE, "Install the weather extra to run weather tests.")
class WeatherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_three_tasks_train_and_reload(self):
        from prola import PROLAStep
        from prola.data.synthetic import SyntheticDataset
        from prola.models.aurora import build_model
        from prola.tasks import get_task, loss
        from prola.training import read_config

        for name in ("pswd", "cerra", "precipitation"):
            with self.subTest(task=name):
                config = read_config(
                    Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
                )
                task = get_task(name)
                dataset = SyntheticDataset(task)
                model = build_model(config, task, dataset.grid, tiny=True)
                sample = next(iter(DataLoader(dataset, batch_size=2)))
                output = model(sample)
                value = loss(output, sample, task)
                self.assertTrue(torch.isfinite(value))
                value.backward()
                PROLAStep(
                    model,
                    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4),
                ).step(1.0)
                restored = build_model(config, task, dataset.grid, tiny=True, initialize=False)
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    for key, expected in model(sample).items():
                        torch.testing.assert_close(restored(sample)[key], expected)

    def test_task_metric_batch_invariance(self):
        from prola.data.synthetic import SyntheticDataset, SyntheticClimatology
        from prola.metrics.weather import WeatherMetrics
        from prola.metrics.precipitation import PrecipitationMetrics
        from prola.tasks import get_task

        for name in ("pswd", "cerra", "precipitation"):
            task = get_task(name)
            data = SyntheticDataset(task, count=3)
            results = []
            for batch_size in (1, 2, 3):
                metric = (
                    PrecipitationMetrics(task, data.grid, SyntheticClimatology(data.grid))
                    if name == "precipitation"
                    else WeatherMetrics(task, data.grid)
                )
                for sample in DataLoader(data, batch_size=batch_size):
                    if name == "precipitation":
                        pred = {"precipitation": sample["target_precipitation"] + 0.0002}
                    else:
                        pred = dict(
                            surface=sample["target_surface"] + 1, upper=sample["target_upper"] + 1
                        )
                    metric.update(pred, sample)
                results.append(metric.result())
            for key in results[0]:
                np.testing.assert_allclose(
                    [r[key] for r in results], results[0][key], rtol=1e-6, atol=1e-8
                )

    def test_split_bounds(self):
        from prola.data.common import anchors, timestamp

        start, end = "2018-01-01", "2019-01-01"
        for lag, leads in ((6, (1, 2, 3, 4, 5, 6)), (6, (3, 6)), (12, (12, 24, 36))):
            keys = anchors(start, end, lag, leads, lag)
            self.assertGreaterEqual(keys[0] - lag * 3600, timestamp(start))
            self.assertLess(keys[-1] + max(leads) * 3600, timestamp(end))

    def test_netcdf_dimension_order(self):
        import xarray as xr
        from prola.data.common import ERA5Store, SURFACE, timestamp

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "2m_temperature"
            folder.mkdir()
            values = np.arange(6, dtype="float32").reshape(1, 3, 2)
            data = xr.Dataset(
                {"t2m": (("time", "longitude", "latitude"), values)},
                coords=dict(
                    time=[np.datetime64("2018-01-01")], longitude=[0, 120, 240], latitude=[-45, 45]
                ),
            )
            data.to_netcdf(folder / "2m_temperature_2018_5.625deg.nc")
            store = ERA5Store(root, "5.625")
            field = store.array(timestamp("2018-01-01"), SURFACE[0])
            self.assertEqual(field.dims, ("latitude", "longitude"))
            np.testing.assert_array_equal(field.values, values[0].T[::-1])
            store.close()

    def test_mean_per_case_rmse(self):
        from prola.metrics.weather import WeatherMetrics, VARIABLES
        from prola.tasks import get_task

        task = get_task("cerra")
        metric = WeatherMetrics(task, dict(latitude=np.array([60, 0])))
        surface = torch.zeros(2, 2, 3, 2, 4)
        upper = torch.zeros(2, 2, 5, 2, 2, 4)
        errors = torch.tensor([1.0, 3.0])
        sample = dict(target_surface=surface, target_upper=upper, anchor_time=torch.tensor([0, 1]))
        metric.update(
            dict(
                surface=surface + errors.view(2, 1, 1, 1, 1),
                upper=upper + errors.view(2, 1, 1, 1, 1, 1),
            ),
            sample,
        )
        for variable in VARIABLES:
            self.assertAlmostEqual(metric.result()[f"{variable}/rmse/3h"], 2.0)


if __name__ == "__main__":
    unittest.main()
