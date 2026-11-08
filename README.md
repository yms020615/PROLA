# PROLA

**Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling**

Minseo Yoon · Minseong Bae · Sojin Lee · Hyunwoo J. Kim · KAIST

PROLA combines a principal-initialized branch with an orthogonal branch under a fixed low-rank budget, and rescales optimizer updates using gradient signals. This repository includes PROLA and three Aurora-based weather tasks.

| Task | Inputs | Targets |
| --- | --- | --- |
| PSWD | Two 5.625° global states, 6h apart | 1.5° fields at 1–6h |
| ERA5-to-CERRA | Two coarse ERA5 states, 6h apart | Regional CERRA fields at 3h and 6h |
| Regional precipitation | Two regional ERA5 states, 12h apart | 6h precipitation totals ending at 12h, 24h, and 36h |

## Quick start

Tested with Python 3.10 and PyTorch 2.5.1. From the repository root:

```bash
pip install -e .
python examples/minimal_prola.py
```

The CPU example checks initial function preservation, performs three updates, and verifies checkpoint restoration. It needs no weather data or model download.

## Weather examples

Install the weather dependencies:

```bash
pip install -e '.[weather]'
```

Run a small training and evaluation example with synthetic data:

```bash
python -m prola.train --config configs/pswd.yaml --smoke-test --device cpu --output-dir runs/pswd_demo
python -m prola.evaluate --run-dir runs/pswd_demo --device cpu
```

Use `configs/cerra.yaml` or `configs/precipitation.yaml` and a different output directory to check the other tasks. These examples use a small randomly initialized backbone; their metrics are functional checks.

## Training and evaluation

Prepare the data described in [Data](docs/data.md) and the Aurora `aurora-0.25-pretrained.ckpt` checkpoint from [Aurora](https://github.com/microsoft/aurora).

```bash
python -m prola.train --config configs/pswd.yaml --data-root data/pswd --backbone-checkpoint checkpoints/aurora-0.25-pretrained.ckpt --output-dir runs/pswd_seed0 --seed 0
python -m prola.evaluate --run-dir runs/pswd_seed0 --data-root data/pswd --split test
```

For multiple GPUs, use the same training command through `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m prola.train --config configs/pswd.yaml --data-root data/pswd --backbone-checkpoint checkpoints/aurora-0.25-pretrained.ckpt --output-dir runs/pswd_seed0_ddp
```

Change the config and data root for CERRA or precipitation. GPU selection follows `CUDA_VISIBLE_DEVICES`. Gradient accumulation preserves the configured effective batch size. Training saves `config.json`, `train.jsonl`, and `last.pt`; evaluation writes per-lead metrics. Each run requires a new output directory.

| Configuration | Epochs | Effective batch | Learning rate |
| --- | ---: | ---: | ---: |
| PSWD | 10 | 32 | 5e-4 |
| CERRA | 10 | 8 | 5e-4 |
| Precipitation | 10 | 32 | 3.75e-4 |

All three presets use principal rank 4, orthogonal rank 4, alpha 1, and enabled update scaling. The forward coefficient is 1/4 in each branch, matching the archived experiments. The global comparison uses seeds 0, 1, and 2; the regional comparisons use seed 0. Evaluate the final checkpoint.

Real-data checkpoints contain the adapter and task-head state. Evaluation also requires the original pretrained checkpoint; `--backbone-checkpoint` can provide its new location.

## Code

| Location | Contents |
| --- | --- |
| `src/prola/adapters/` | PROLA layers, injection, and optimizer-update scaling |
| `src/prola/models/` | Correction heads, prediction pipeline, and bundled Aurora implementation |
| `src/prola/tasks/` | Variable sets, forecast leads, and losses |
| `src/prola/data/` | Dataset readers |
| `src/prola/metrics/` | RMSE, ACC, and SEEPS |
| `src/prola/training.py` | Training, checkpoint, and evaluation utilities |
| `configs/` | The three experiment presets |

PSWD and CERRA share the scaffold-and-correct pipeline. Precipitation uses its own output heads. Adapters target the backbone attention projections.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@misc{yoon2026prola,
  title={PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling},
  author={Yoon, Minseo and Bae, Minseong and Lee, Sojin and Kim, Hyunwoo J.},
  year={2026}
}
```

The weather implementation builds on [Aurora](https://github.com/microsoft/aurora). See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES).
