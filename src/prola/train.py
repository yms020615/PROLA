import argparse
from pathlib import Path

import torch.distributed as dist

from prola.data import build_dataset
from prola.models.aurora import build_model
from prola.tasks import get_task
from prola.training import read_config, setup, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if Path(args.output_dir).exists():
        parser.error("The output directory already exists; choose a new directory.")
    if not args.smoke_test and (not args.data_root or not args.backbone_checkpoint):
        parser.error("Real-data training requires --data-root and --backbone-checkpoint.")
    device, rank, world = setup(config["seed"], args.device)
    task = get_task(config["task"])
    if args.smoke_test:
        from prola.data.synthetic import SyntheticDataset

        dataset = SyntheticDataset(task, count=4 * world)
        config["training"].update(
            epochs=1, batch_size=1, effective_batch_size=2 * world, warmup_epochs=0
        )
    else:
        dataset = build_dataset(config, task, args.data_root, "train")
    model = build_model(
        config,
        task,
        dataset.grid,
        checkpoint=args.backbone_checkpoint,
        tiny=args.smoke_test,
        device=device,
    )
    train(
        model,
        dataset,
        task,
        config,
        args.output_dir,
        device,
        rank,
        world,
        args.workers,
        args.backbone_checkpoint,
        args.smoke_test,
    )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
