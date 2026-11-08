import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from prola.data import build_dataset
from prola.metrics import build_metrics
from prola.models.aurora import build_model
from prola.tasks import get_task
from prola.training import setup, evaluate, load_adaptation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", dest="run", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    run = Path(args.run)
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    config, tiny = saved["config"], saved["tiny"]
    if not tiny and not args.data_root:
        parser.error("Real-data evaluation requires --data-root.")
    device, rank, world = setup(config["seed"], args.device)
    task = get_task(config["task"])
    if tiny:
        from prola.data.synthetic import SyntheticDataset

        dataset = SyntheticDataset(task)
    else:
        dataset = build_dataset(config, task, args.data_root, args.split)
    model = build_model(
        config,
        task,
        dataset.grid,
        checkpoint=args.backbone_checkpoint or saved["pretrained"],
        tiny=tiny,
        initialize=False,
        device=device,
    )
    if tiny:
        model.load_state_dict(saved["model"], strict=True)
    else:
        load_adaptation(model, saved["model"])
    metrics = build_metrics(config, task, dataset.grid, args.data_root, synthetic=tiny)
    results = evaluate(model, dataset, metrics, device, rank, world, args.batch_size, args.workers)
    if rank == 0:
        payload = dict(
            task=task["name"],
            split="synthetic" if tiny else args.split,
            samples=metrics.count,
            metrics=results,
        )
        (run / f"{payload['split']}_metrics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(payload, indent=2, allow_nan=False))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
