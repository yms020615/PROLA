from importlib import import_module


def get_task(name):
    if name not in {"pswd", "cerra", "precipitation"}:
        raise ValueError(f"Unknown task: {name}")
    return dict(import_module(f"prola.tasks.{name}").TASK)


def loss(output, sample, task):
    import torch
    from prola.models.pipeline import normalize_surface, normalize_upper
    from prola.models._aurora.normalisation import normalise_surf_var

    if task["name"] == "precipitation":
        target = normalise_surf_var(sample["target_precipitation"], "tp", None)
        return (output["precipitation"] - target).abs().mean()
    surface = normalize_surface(sample["target_surface"], task["surface"])
    upper = normalize_upper(sample["target_upper"], task["upper"], tuple(task["levels"]))
    s_error = (output["surface"] - surface).abs().mean(dim=(0, 1, 3, 4))
    u_error = (output["upper"] - upper).abs().mean(dim=(0, 1, 3, 4, 5))
    s_weights = torch.tensor([3.0, 0.77, 0.66], device=s_error.device)
    u_weights = torch.tensor([1.5, 0.77, 0.54, 0.6, 3.0], device=u_error.device)
    return (0.25 * (s_error * s_weights).sum() + (u_error * u_weights).sum()) / 8
