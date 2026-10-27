"""
Utility Functions for Face Anti-Spoofing System
Author: Team STT 12
"""

import os
import sys
import random
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.is_absolute():
        # Try relative to project root or current working dir
        candidates = [
            path,
            Path(__file__).resolve().parent.parent / config_path,
            Path.cwd() / config_path,
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Sets random seeds across random, numpy, and torch for experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "cuda"):
    """
    Returns appropriate torch device (cuda if available and requested, else cpu).
    """
    try:
        import torch
        if preference.lower() == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    except ImportError:
        return "cpu"


def setup_logger(name: str = "FAS", log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth",
    best_filename: str = "best_model.pth"
):
    """Saves model checkpoint and optionally saves as best model."""
    import torch
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filepath = ckpt_dir / filename
    torch.save(state, str(filepath))

    if is_best:
        best_filepath = ckpt_dir / best_filename
        torch.save(state, str(best_filepath))


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu"
) -> Dict[str, Any]:
    """Loads weights and state from checkpoint file."""
    import torch
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(str(path), map_location=device)

    # Support state dict wrapped or raw model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logging.warning(f"Failed to load scheduler state: {e}")

    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class FPSCounter:
    """Helper class to track and smooth FPS in realtime demo."""
    def __init__(self, avg_window: int = 30):
        self.avg_window = avg_window
        self.frame_times = []
        self.prev_time = time.time()

    def tick(self) -> float:
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0:
            self.frame_times.append(1.0 / delta)
            if len(self.frame_times) > self.avg_window:
                self.frame_times.pop(0)

        if self.frame_times:
            return sum(self.frame_times) / len(self.frame_times)
        return 0.0
