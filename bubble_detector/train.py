import argparse
import copy
import glob
import os
import os.path as osp
import time
from enum import Enum
from pathlib import Path

import mmcv
import yaml
from mmcv import Config
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    collect_env,
    get_device,
    get_root_logger,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)


class Backbones(str, Enum):
    swin_b = "swin-b-p4-w7"
    swin_t = "swin-t-p4-w7"
    Swin_s = "swin-s-p4-w7"
    resnet_101 = "r101"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="path to coco format dataset", type=Path)
    parser.add_argument("--name", help="experiment name", type=str, default="")
    parser.add_argument("--epochs", help="number of epochs to train for", type=int, default=80)
    parser.add_argument(
        "--backbone",
        help="which backbone to use",
        choices=[b.name for b in Backbones],
        default=Backbones.swin_b.name,
    )
    parser.add_argument("--distributed", help="distributed learning", action="store_true")

    return parser.parse_args()


def generate_dist_path(name: str) -> Path:
    name = name if name else "train"
    for i in range(1000):
        dist_path = Path(f"runs/{name}{f'_{i}' if i > 0 else ''}")
        if not dist_path.exists():
            dist_path.mkdir(parents=True, exist_ok=True)
            return dist_path
    raise RuntimeError("Could not generate dist_path")


def rel2abs(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def create_config(dataset_path: Path, name: str, epochs: int, backbone: Backbones) -> Path:
    here = Path(__file__).parent.absolute()
    config_path = here / f"configs/cascade_mask_rcnn_{backbone}_fpn.py"
    cfg = mmcv.Config.fromfile(config_path)

    # Dataset Settings
    with open(dataset_path / "data.yaml", "r") as f:
        dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    cfg.dataset_type = "COCODataset"
    cfg.data.train.ann_file = rel2abs(Path(dataset_info["train_labels"]), dataset_path).as_posix()
    cfg.data.train.img_prefix = rel2abs(Path(dataset_info["train"]), dataset_path).as_posix()
    cfg.data.train.classes = ("bubble",)

    cfg.data.val.ann_file = rel2abs(Path(dataset_info["val_labels"]), dataset_path).as_posix()
    cfg.data.val.img_prefix = rel2abs(Path(dataset_info["val"]), dataset_path).as_posix()
    cfg.data.val.classes = ("bubble",)

    # Epochs Settings
    cfg.runner.max_epochs = epochs

    # Model Path Settings
    dist_path = generate_dist_path(name)
    cfg.work_dir = dist_path.as_posix()
    config_path = dist_path / "config.py"
    cfg.dump(config_path.as_posix())

    return config_path


def train(config_path: str, distributed: bool = False) -> None:
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    setup_multi_processes(cfg)

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    cfg.device = get_device() if False else "cpu"
    print(f"Using device: {cfg.device}")
    seed = cfg.seed
    logger.info(f"Set random seed to {seed}, distributed: {distributed}")
    init_random_seed(device=cfg.device)
    set_random_seed(seed)
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(config_path)

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"))
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        assert "val" in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            "pipeline", cfg.data.train.dataset.get("pipeline")
        )
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )

    checkpoints = glob.glob(osp.join(osp.dirname(config_path), "*.pth"))
    for checkpoint in checkpoints:
        filename = osp.basename(checkpoint)
        if "latest" not in filename and "best" not in filename:
            os.remove(checkpoint)


def main() -> None:
    args = get_args()
    config_path = create_config(
        args.dataset_path, args.name, args.epochs, Backbones[args.backbone]
    )
    train(config_path, args.distributed)


if __name__ == "__main__":
    main()
