import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from prola.adapters import PROLALinear, PROLAStep
from prola.tasks import loss


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config = json.loads(json.dumps(config, default=str))
    if not config["adapter"]["update_scaling"]:
        raise ValueError("The released PROLA configurations require update scaling.")
    return config


def setup(seed, device):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(local if world > 1 else device.index or 0)
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", 1))))
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device, rank, world


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            dict(params=decay, weight_decay=config["weight_decay"]),
            dict(params=no_decay, weight_decay=0),
        ],
        lr=config["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def move(sample, device):
    return {k: v.to(device, non_blocking=True) for k, v in sample.items()}


def learning_rate(step, total, warmup, config):
    if step < warmup:
        return config["warmup_lr"] + (config["lr"] - config["warmup_lr"]) * step / max(
            1, warmup - 1
        )
    fraction = (step - warmup) / max(1, total - warmup)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (
        1 + math.cos(math.pi * fraction)
    )


def adaptation_state(model):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    buffers = set()
    for name, layer in model.named_modules():
        if isinstance(layer, PROLALinear):
            buffers.update(f"{name}.{key}" for key, _ in layer.named_buffers())
    return {
        k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable or k in buffers
    }


def load_adaptation(model, state):
    if any(bool(layer.initialized) for layer in model.modules() if isinstance(layer, PROLALinear)):
        raise ValueError("Load adaptation into a fresh model built with initialize=False.")
    expected = set(adaptation_state(model))
    if set(state) != expected:
        raise ValueError(
            f"Adapter checkpoint keys do not match: missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}"
        )
    model.load_state_dict(state, strict=False)
    for layer in model.modules():
        if isinstance(layer, PROLALinear):
            layer.residualize_loaded_base()


def train(model, dataset, task, config, output, device, rank, world, workers, pretrained, tiny):
    cfg = config["training"]
    micro = cfg["batch_size"]
    if cfg["effective_batch_size"] % (micro * world):
        raise ValueError("Effective batch size must be divisible by batch_size × world_size.")
    accumulate = cfg["effective_batch_size"] // (micro * world)
    if accumulate < 1:
        raise ValueError("Effective batch size is smaller than the distributed micro-batch.")
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world, rank=rank, seed=config["seed"], drop_last=True
        )
        if world > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=micro,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    steps = len(loader) // accumulate
    if not steps:
        raise ValueError("The training dataset is smaller than one effective batch.")
    base = model.to(device)
    optimizer = optimizer_for(base, cfg)
    stepper = PROLAStep(base, optimizer)
    model = (
        DistributedDataParallel(
            base,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if world > 1
        else base
    )
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    total, warmup = steps * cfg["epochs"], steps * cfg["warmup_epochs"]
    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for index, sample in enumerate(loader):
            if index >= steps * accumulate:
                break
            step = epoch * steps + index // accumulate
            if index % accumulate == 0:
                lr = learning_rate(step, total, warmup, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            boundary = (index + 1) % accumulate == 0
            context = model.no_sync() if world > 1 and not boundary else nullcontext()
            with context:
                sample = move(sample, device)
                value = loss(model(sample), sample, task)
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite training loss.")
                (value / accumulate).backward()
            loss_sum += value.detach()
            if boundary:
                stepper.step(cfg["clip_grad"])
                optimizer.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(loss_sum)
        mean = float(loss_sum / (steps * accumulate * world))
        if rank == 0:
            record = dict(epoch=epoch + 1, loss=mean, lr=lr)
            print(json.dumps(record), flush=True)
            with (output / "train.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            state = (
                {k: v.detach().cpu() for k, v in base.state_dict().items()}
                if tiny
                else adaptation_state(base)
            )
            checkpoint = dict(
                model=state,
                config=config,
                epoch=epoch + 1,
                tiny=tiny,
                pretrained=str(Path(pretrained).resolve()) if pretrained else None,
            )
            torch.save(checkpoint, output / "last.pt.tmp")
            (output / "last.pt.tmp").replace(output / "last.pt")
        if world > 1:
            dist.barrier()


@torch.no_grad()
def evaluate(model, dataset, metrics, device, rank=0, world=1, batch_size=1, workers=0):
    loader = DataLoader(
        Subset(dataset, range(rank, len(dataset), world)),
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval().to(device)
    for sample in loader:
        sample = move(sample, device)
        metrics.update(model.denormalize(model(sample)), sample)
    if world > 1:
        totals = torch.as_tensor(metrics.totals, device=device)
        count = torch.tensor(metrics.count, device=device, dtype=torch.long)
        dist.all_reduce(totals)
        dist.all_reduce(count)
        metrics.totals, metrics.count = totals.cpu().numpy(), int(count)
    return metrics.result()
