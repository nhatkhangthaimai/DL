#!/usr/bin/env python3
"""
Deep Learning Training Pipeline for Face Anti-Spoofing
Supports ResNet34 + LBP Feature Fusion and Pure ResNet34 Baseline
Author: Team STT 12
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np

from src.utils import (
    load_config,
    set_seed,
    get_device,
    setup_logger,
    save_checkpoint,
    AverageMeter
)
from src.dataset import create_dataloaders
from src.model import build_model
from src.metrics import compute_fas_metrics, plot_training_curves, find_optimal_threshold


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "fusion"
) -> Tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for rgb_imgs, lbp_feats, labels, _ in pbar:
        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if model_type == "baseline":
            logits = model(rgb_imgs)
        else:
            lbp_feats = lbp_feats.to(device, non_blocking=True)
            logits = model(rgb_imgs, lbp_feats)

        # Check for NaN / Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf detected in model forward outputs!")

        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN or Inf detected in training loss!")

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        loss_meter.update(loss.item(), rgb_imgs.size(0))
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{100.0 * correct / max(1, total):.1f}%"})

    acc = 100.0 * correct / max(1, total)
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "fusion"
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates validation performance and computes full FAS metrics."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(val_loader, desc="  Validation", leave=False)
    for rgb_imgs, lbp_feats, labels, _ in pbar:
        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if model_type == "baseline":
            logits = model(rgb_imgs)
        else:
            lbp_feats = lbp_feats.to(device, non_blocking=True)
            logits = model(rgb_imgs, lbp_feats)

        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), rgb_imgs.size(0))

        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of Spoof (class 1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_fas_metrics(y_true, y_pred, y_prob)
    return loss_meter.avg, metrics


def train_model(
    config_path: str = "configs/config.yaml",
    model_type: str = "fusion",
    override_epochs: int = None,
    override_batch_size: int = None,
    override_lr: float = None,
    override_device: str = None
):
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    pref_device = override_device or config.get("project", {}).get("device", "cuda")
    device = get_device(pref_device)

    log_file = f"logs/train_{model_type}.log"
    logger = setup_logger(f"Train_{model_type.upper()}", log_file)

    logger.info("=" * 65)
    logger.info(f"   STARTING TRAINING: MODEL = {model_type.upper()}")
    logger.info(f"   Target Device: {device}")
    logger.info("=" * 65)

    base_dir = Path(__file__).resolve().parent
    metadata_csv = base_dir / config.get("paths", {}).get("processed_dir", "processed") / "metadata.csv"

    if not metadata_csv.exists():
        logger.error(f"Metadata file not found at: {metadata_csv}. Please run prepare_dataset.py first!")
        sys.exit(1)

    # Overrides if specified
    if override_batch_size:
        config["training"]["batch_size"] = override_batch_size
    if override_epochs:
        config["training"]["epochs"] = override_epochs
    if override_lr:
        config["training"]["learning_rate"] = override_lr

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(str(metadata_csv), config)
    logger.info(f"Dataset Loaded:")
    logger.info(f"  - Train Batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    logger.info(f"  - Val Batches:   {len(val_loader)} ({len(val_loader.dataset)} samples)")
    logger.info(f"  - Test Batches:  {len(test_loader)} ({len(test_loader.dataset)} samples)")
    logger.info(f"  - Computed Class Weights: Live={class_weights[0]:.2f}, Spoof={class_weights[1]:.2f}")

    # Build model
    model = build_model(config, model_type=model_type).to(device)

    # Loss with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Hyperparameters
    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 20)
    lr = t_cfg.get("learning_rate", 0.0001)
    backbone_lr = t_cfg.get("backbone_lr", 0.00002)
    weight_decay = t_cfg.get("weight_decay", 0.0001)
    patience = t_cfg.get("early_stopping_patience", 6)
    freeze_epochs = t_cfg.get("freeze_backbone_epochs", 2)

    # Staged Optimizer setup (Transfer Learning & Fine-tuning)
    if freeze_epochs > 0:
        logger.info(f"Initial warm-up: Freezing ResNet34 backbone for {freeze_epochs} epochs.")
        model.freeze_backbone()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint and result paths
    ckpt_dir = base_dir / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_name = f"best_{model_type}_model.pth"
    latest_ckpt_name = f"checkpoint_{model_type}.pth"

    results_dir = base_dir / config.get("paths", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_apcer": [],
        "val_bpcer": [],
        "val_acer": [],
        "val_auc": []
    }

    best_val_acer = float("inf")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warm-up epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            logger.info("[*] Unfreezing ResNet34 backbone for end-to-end fine-tuning with differential LR!")
            model.unfreeze_backbone()
            # Differential learning rates: smaller LR for pretrained backbone, higher for fusion head
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "conv1" in name or "bn1" in name or "layer" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = AdamW([
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr}
            ], weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

        # Train epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_type=model_type
        )

        # Validation epoch
        val_loss, val_m = evaluate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            model_type=model_type
        )

        scheduler.step()
        epoch_sec = time.time() - epoch_start

        val_acc = val_m["accuracy"] * 100.0
        val_acer = val_m["acer"]
        val_apcer = val_m["apcer"]
        val_bpcer = val_m["bpcer"]
        val_auc = val_m.get("auc", 0.0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_apcer"].append(val_apcer)
        history["val_bpcer"].append(val_bpcer)
        history["val_acer"].append(val_acer)
        history["val_auc"].append(val_auc)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
            f"ACER: {val_acer:.4f} (APCER: {val_apcer:.4f}, BPCER: {val_bpcer:.4f}), AUC: {val_auc:.4f}"
        )

        # Check for best model (based on ACER, lower is better)
        is_best = val_acer < best_val_acer
        if is_best:
            best_val_acer = val_acer
            patience_counter = 0
            logger.info(f"  --> [*] New Best ACER: {best_val_acer:.4f}! Saving checkpoint to {best_ckpt_name}")
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            state={
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acer": val_acer,
                "val_metrics": val_m,
                "config": config
            },
            is_best=is_best,
            checkpoint_dir=str(ckpt_dir),
            filename=latest_ckpt_name,
            best_filename=best_ckpt_name
        )

        # Early Stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs without ACER improvement.")
            break

    total_training_sec = time.time() - start_time
    logger.info("=" * 65)
    logger.info(f"TRAINING COMPLETED in {total_training_sec / 60:.2f} minutes.")
    logger.info(f"Best Validation ACER: {best_val_acer:.4f}")
    logger.info("=" * 65)

    # Save training history JSON
    history_file = results_dir / f"history_{model_type}.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # Export curves
    curve_img = results_dir / f"training_curves_{model_type}.png"
    plot_training_curves(history, str(curve_img))
    logger.info(f"Training curves saved to: {curve_img}")

    return history, best_val_acer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face Anti-Spoofing Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "baseline"], help="Model type: 'fusion' or 'baseline'")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        model_type=args.model,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        override_device=args.device
    )
