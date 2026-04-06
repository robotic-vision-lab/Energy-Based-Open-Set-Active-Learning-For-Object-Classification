"""Generic utilities for EB-OSAL.

This module is intentionally method-agnostic. It only contains shared helpers
that are useful across training, evaluation, logging, checkpointing, and basic
configuration handling.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------
def get_device(device_name: Optional[str] = None) -> torch.device:
    """Resolve a torch device from config or system availability."""
    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_name = device_name.lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def move_to_device(batch, device: torch.device):
    """Recursively move tensors inside a nested batch to device."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch

# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    """Create the standard output directory tree for the project."""
    root = ensure_dir(output_dir)
    checkpoints = ensure_dir(root / "checkpoints")
    logs = ensure_dir(root / "logs")
    results = ensure_dir(root / "results")
    return {
        "root": root,
        "checkpoints": checkpoints,
        "logs": logs,
        "results": results,
    }


def timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_experiment_name(config: Dict[str, Any]) -> str:
    """Create a compact experiment name from config values."""
    dataset_name = config.get("dataset", {}).get("name", "dataset")
    backbone = config.get("model", {}).get("backbone", "model")
    mismatch_ratio = config.get("dataset", {}).get("mismatch_ratio", "na")
    seed = config.get("seed", "na")
    return f"{dataset_name}_{backbone}_mr{mismatch_ratio}_seed{seed}_{timestamp_string()}"

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(default_config_path: str | Path, override_config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load default config and optionally merge a dataset-specific override."""
    config = load_yaml(default_config_path)
    if override_config_path is not None:
        override = load_yaml(override_config_path)
        config = deep_update(config, override)
    return dict(config)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(
    name: str = "eb_osal",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console logger and optionally attach a file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))

# -----------------------------------------------------------------------------
# Metric tracking
# -----------------------------------------------------------------------------
class AverageMeter:
    """Track the running average of a scalar metric."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def as_dict(self) -> Dict[str, float]:
        return {
            "value": self.val,
            "avg": self.avg,
            "sum": self.sum,
            "count": float(self.count),
        }

class MetricTracker:
    """A small container for multiple AverageMeter objects."""
    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def as_dict(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1,)) -> List[float]:
    """Compute top-k accuracies for a batch."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return results


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------
def to_serializable(obj: Any) -> Any:
    """Convert nested objects into JSON-serializable forms."""
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])