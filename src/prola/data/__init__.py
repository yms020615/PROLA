from pathlib import Path


def build_dataset(config, task, root, split):
    if task["name"] == "pswd":
        from .pswd import PSWDDataset as Dataset
    elif task["name"] == "cerra":
        from .cerra import CERRADataset as Dataset
    else:
        from .precipitation import PrecipitationDataset as Dataset
    return Dataset(Path(root), config["data"], split, task)
