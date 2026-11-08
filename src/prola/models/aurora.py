from datetime import timedelta

from prola.adapters import inject_prola
from ._aurora import Aurora
from .pipeline import WeatherModel


TINY = dict(
    embed_dim=32,
    encoder_depths=(1, 1),
    decoder_depths=(1, 1),
    encoder_num_heads=(2, 4),
    decoder_num_heads=(4, 2),
    num_heads=4,
    latent_levels=2,
    window_size=(2, 2, 2),
)


def build_model(config, task, grid, checkpoint=None, tiny=False, initialize=True, device=None):
    options = dict(
        surf_vars=tuple(task["surface"]),
        atmos_vars=tuple(task["upper"]),
        timestep=timedelta(hours=task["input_lag"]),
        autocast=config["training"].get("bf16", True),
    )
    if task["name"] == "precipitation":
        options.update(out_surf_vars=tuple(f"tp_{h}" for h in task["leads"]), out_atmos_vars=())
    backbone = Aurora(**options, **(TINY if tiny else {}))
    if checkpoint:
        backbone.load_checkpoint_local(str(checkpoint), strict=False)
    elif not tiny:
        raise ValueError("A pretrained Aurora checkpoint is required.")
    backbone.requires_grad_(False)
    if device is not None:
        backbone.to(device)
    adapter = dict(config["adapter"])
    adapter.pop("update_scaling")
    if tiny:
        adapter["svd_iterations"] = 0
    inject_prola(backbone.backbone, initialize=initialize, **adapter)
    if task["name"] == "precipitation":
        backbone.decoder.surf_heads.requires_grad_(True)
    if not tiny:
        backbone.configure_activation_checkpointing()
    model = WeatherModel(backbone, task, grid)
    return model.to(device) if device is not None else model
