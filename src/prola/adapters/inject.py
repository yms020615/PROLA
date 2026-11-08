from torch import nn

from .linear import PROLALinear


def inject_prola(model: nn.Module, target_modules=("qkv", "proj"), **options):
    replacements = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in target_modules
    ]
    if not replacements:
        raise ValueError(f"No linear layers match {target_modules}.")
    for name, layer in replacements:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, PROLALinear(layer, **options))
    return [name for name, _ in replacements]
