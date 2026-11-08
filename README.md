<div align="center">

# PROLA

### Principal-Orthogonal Low-rank Adaptation<br>for Predictive Spatiotemporal Weather Downscaling

Minseo Yoon · Minseong Bae · Sojin Lee · Hyunwoo J. Kim

**KAIST · NeurIPS 2026**

[Overview](#overview) · [Results](#results) · [Quick start](#quick-start) · [Data](docs/data.md) · [Citation](#citation)

</div>

**From coarse weather states to fine-scale future predictions.** PROLA adapts pretrained weather models through complementary principal and orthogonal low-rank branches.

<p align="center">
  <img src="docs/assets/cerra_t2m_6h.png" width="1000" alt="ERA5-to-CERRA temperature predictions at 6h: LoRA, PROLA, and ground truth over Central Europe.">
  <br>
  <em>ERA5-to-CERRA downscaling: 2m temperature at 6h, valid on 4 February 2018 at 12:00 UTC.</em>
</p>

## Overview

**PSWD** combines spatial refinement, temporal refinement, and future prediction. A scaffold-and-correct pipeline pairs a pretrained weather backbone with lightweight spatial and temporal correction heads.

**PROLA** splits a fixed rank budget between pretrained principal directions and their orthogonal complement, then uses gradient signals to scale optimizer updates. The main PSWD model trains **3.92M parameters**, less than **0.4%** of the full backbone size.

<p align="center">
  <img src="docs/assets/prola_overview.png" width="1000" alt="PROLA architecture: principal and orthogonal low-rank branches, followed by gradient-guided scaling of optimizer updates.">
</p>

This repository covers three weather tasks:

| Task | Prediction |
| --- | --- |
| **PSWD** | 5.625° global inputs → 1.5° hourly forecasts at 1–6h |
| **ERA5-to-CERRA** | Coarse ERA5 inputs → regional CERRA fields at 3h and 6h |
| **Regional precipitation** | Regional ERA5 inputs → 6h rainfall totals ending at 12h, 24h, and 36h |

## Results

Main PSWD results from the paper: **3h latitude-weighted RMSE ↓**, averaged over three seeds.

| Method | Z500 | T850 | T2M | U10 | V10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 54.72 | 0.898 | 1.199 | 1.219 | 1.181 |
| DoRA | 54.32 | 0.893 | 1.191 | 1.209 | 1.173 |
| AdaLoRA | 53.40 | 0.884 | 1.176 | 1.175 | 1.158 |
| PiSSA | 54.55 | 0.893 | 1.198 | 1.215 | 1.175 |
| OPLoRA | 54.94 | 0.899 | 1.204 | 1.223 | 1.185 |
| **PROLA** | **50.58** | **0.864** | **1.129** | **1.143** | **1.120** |

## Quick start

Tested with Python 3.10 and PyTorch 2.5.1. From the repository root:

```bash
pip install -e '.[weather]'
python examples/minimal_prola.py
```

The example runs PROLA updates on CPU without downloading data or model weights.

<details>
<summary>Try a weather example with synthetic data</summary>

```bash
python -m prola.train --config configs/pswd.yaml \
  --smoke-test --device cpu --output-dir runs/pswd_demo
python -m prola.evaluate --run-dir runs/pswd_demo --device cpu
```

This checks training and evaluation with a small randomly initialized backbone.

</details>

## Training and evaluation

Prepare the [data](docs/data.md) and the `aurora-0.25-pretrained.ckpt` weights from [Aurora](https://github.com/microsoft/aurora), then run:

```bash
python -m prola.train --config configs/pswd.yaml \
  --data-root data/pswd \
  --backbone-checkpoint checkpoints/aurora-0.25-pretrained.ckpt \
  --output-dir runs/pswd

python -m prola.evaluate --run-dir runs/pswd \
  --data-root data/pswd --split test
```

For the regional tasks, use [`cerra.yaml`](configs/cerra.yaml) or [`precipitation.yaml`](configs/precipitation.yaml) with the corresponding data and output directories.

## Code

```text
src/prola/
  adapters/  models/  tasks/  data/  metrics/
  train.py   evaluate.py   training.py
configs/
examples/
tests/
```

## Citation

```bibtex
@inproceedings{yoon2026prola,
  title={{PROLA}: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling},
  author={Yoon, Minseo and Bae, Minseong and Lee, Sojin and Kim, Hyunwoo J.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments

Our weather implementation builds on [Aurora](https://github.com/microsoft/aurora). See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES).
