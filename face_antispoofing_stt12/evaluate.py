#!/usr/bin/env python3
"""
Model Evaluation and Comparison Pipeline for Face Anti-Spoofing
Computes frame-level and video-level metrics (Accuracy, Precision, Recall, F1, APCER, BPCER, ACER, AUC)
Author: Team STT 12
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils import load_config, get_device, setup_logger, load_checkpoint
from src.dataset import create_dataloaders, FaceAntiSpoofingDataset
from src.model import build_model
from src.metrics import (
    compute_fas_metrics,
    compute_video_level_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison
)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_type: str = "fusion"
) -> pd.DataFrame:
    """
    Runs model inference over an entire DataLoader and collects frame-by-frame predictions and metadata.
    """
    model.eval()
    records = []

    pbar = tqdm(data_loader, desc=f"Evaluating {model_type.upper()}")
    for rgb_imgs, lbp_feats, labels, meta in pbar:
        rgb_imgs = rgb_imgs.to(device, non_blocking=True)

        if model_type == "baseline":
            logits = model(rgb_imgs)
        else:
            lbp_feats = lbp_feats.to(device, non_blocking=True)
            logits = model(rgb_imgs, lbp_feats)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        batch_size = rgb_imgs.size(0)
        for i in range(batch_size):
            records.append({
                "image_path": meta["image_path"][i],
                "video_name": meta["video_name"][i],
                "subject_id": meta["subject_id"][i],
                "label": int(labels[i].item()),
                "label_name": meta["label_name"][i],
                "device": meta["device"][i],
                "attack_type": meta["attack_type"][i],
                "frame_idx": int(meta["frame_idx"][i]),
                "split": meta["split"][i],
                "pred_label": int(preds[i]),
                "pred_live_prob": float(probs[i, 0]),
                "pred_spoof_prob": float(probs[i, 1]),
                "pred_prob": float(probs[i, 1])  # Spoof probability
            })

    return pd.DataFrame(records)


def run_evaluation(
    checkpoint_path: str,
    config_path: str = "configs/config.yaml",
    model_type: str = "fusion",
    split: str = "test",
    threshold: float = 0.5
) -> Dict[str, Any]:
    config = load_config(config_path)
    device = get_device(config.get("project", {}).get("device", "cuda"))
    logger = setup_logger(f"Eval_{model_type.upper()}", f"logs/evaluate_{model_type}.log")

    logger.info("=" * 70)
    logger.info(f"   FACE ANTI-SPOOFING EVALUATION: {model_type.upper()}")
    logger.info(f"   Checkpoint: {checkpoint_path}")
    logger.info(f"   Evaluation Split: {split.upper()} | Device: {device}")
    logger.info("=" * 70)

    base_dir = Path(__file__).resolve().parent
    metadata_csv = base_dir / config.get("paths", {}).get("processed_dir", "processed") / "metadata.csv"
    if not metadata_csv.exists():
        logger.error(f"Metadata file not found: {metadata_csv}")
        sys.exit(1)

    # Dataloaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(str(metadata_csv), config)
    target_loader = test_loader if split == "test" else (val_loader if split == "val" else train_loader)

    # Build and load model
    model = build_model(config, model_type=model_type).to(device)
    load_checkpoint(checkpoint_path, model, device=str(device))

    # Optional: Find optimal threshold on validation set if evaluating on test
    if split == "test":
        logger.info("[*] Estimating optimal decision threshold on Validation Set...")
        val_df = evaluate_dataset(model, val_loader, device=device, model_type=model_type)
        optimal_th, min_val_acer = find_optimal_threshold(val_df["label"].values, val_df["pred_prob"].values)
        logger.info(f"    - Optimal Threshold on Val: {optimal_th:.4f} (Val ACER: {min_val_acer:.4f})")
        threshold = optimal_th

    # Evaluate target split
    pred_df = evaluate_dataset(model, target_loader, device=device, model_type=model_type)

    # Frame-level metrics
    y_true = pred_df["label"].values
    y_prob = pred_df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    frame_metrics = compute_fas_metrics(y_true, y_pred, y_prob)

    # Video-level metrics (aggregated by video average probability)
    video_metrics = compute_video_level_metrics(pred_df, threshold=threshold)

    # Print results summary
    logger.info("\n" + "-" * 70)
    logger.info("  FRAME-LEVEL METRICS:")
    logger.info(f"    - Accuracy:        {frame_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"    - Precision:       {frame_metrics['precision']:.4f}")
    logger.info(f"    - Recall:          {frame_metrics['recall']:.4f}")
    logger.info(f"    - F1-Score:        {frame_metrics['f1_score']:.4f}")
    logger.info(f"    - APCER (Spoof Err): {frame_metrics['apcer'] * 100:.2f}% (Spoofs misclassified as Live)")
    logger.info(f"    - BPCER (Live Err):  {frame_metrics['bpcer'] * 100:.2f}% (Live misclassified as Spoof)")
    logger.info(f"    - ACER (Avg Err):    {frame_metrics['acer'] * 100:.2f}%")
    if "auc" in frame_metrics:
        logger.info(f"    - ROC AUC:         {frame_metrics['auc']:.4f}")
    if "eer" in frame_metrics:
        logger.info(f"    - Equal Error Rate (EER): {frame_metrics['eer'] * 100:.2f}%")

    logger.info("\n" + "-" * 70)
    logger.info("  VIDEO-LEVEL METRICS (Aggregated across video frames):")
    logger.info(f"    - Total Videos:    {video_metrics.get('total_videos', 0)}")
    logger.info(f"    - Accuracy:        {video_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"    - Precision:       {video_metrics['precision']:.4f}")
    logger.info(f"    - Recall:          {video_metrics['recall']:.4f}")
    logger.info(f"    - F1-Score:        {video_metrics['f1_score']:.4f}")
    logger.info(f"    - Video APCER:     {video_metrics['apcer'] * 100:.2f}%")
    logger.info(f"    - Video BPCER:     {video_metrics['bpcer'] * 100:.2f}%")
    logger.info(f"    - Video ACER:      {video_metrics['acer'] * 100:.2f}%")
    logger.info("-" * 70)

    # Save artifacts
    results_dir = base_dir / config.get("paths", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions CSV
    pred_csv_path = results_dir / f"predictions_{model_type}_{split}.csv"
    pred_df.to_csv(pred_csv_path, index=False)

    # Save metrics JSON
    metrics_bundle = {
        "model_type": model_type,
        "split": split,
        "threshold": threshold,
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics
    }
    metrics_json_path = results_dir / f"metrics_{model_type}_{split}.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_bundle, f, indent=2)

    # Save Confusion Matrix heatmap
    cm_path = results_dir / f"confusion_matrix_{model_type}_{split}.png"
    cm_arr = np.array(frame_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, str(cm_path), title=f"Confusion Matrix: {model_type.upper()} ({split.upper()})")

    # Save ROC Curve
    if "fpr" in frame_metrics and "tpr" in frame_metrics:
        roc_path = results_dir / f"roc_curve_{model_type}_{split}.png"
        plot_roc_curve(
            frame_metrics["fpr"],
            frame_metrics["tpr"],
            frame_metrics["auc"],
            str(roc_path),
            eer=frame_metrics.get("eer")
        )

    logger.info(f"[+] Results saved to {results_dir}")
    return metrics_bundle


def compare_models(
    baseline_ckpt: str,
    fusion_ckpt: str,
    config_path: str = "configs/config.yaml"
):
    """Evaluates and produces comparative tables and charts for Baseline vs Fusion."""
    logger = setup_logger("ModelComparison", "logs/comparison.log")
    logger.info("=" * 70)
    logger.info("  EVALUATING MODEL COMPARISON (Baseline vs ResNet34 + LBP)")
    logger.info("=" * 70)

    m_baseline = run_evaluation(baseline_ckpt, config_path, model_type="baseline", split="test")
    m_fusion = run_evaluation(fusion_ckpt, config_path, model_type="fusion", split="test")

    b_frame = m_baseline["frame_metrics"]
    f_frame = m_fusion["frame_metrics"]

    comparison_rows = [
        {
            "Model": "ResNet34 (Baseline)",
            "Accuracy": b_frame["accuracy"],
            "Precision": b_frame["precision"],
            "Recall": b_frame["recall"],
            "F1-Score": b_frame["f1_score"],
            "APCER": b_frame["apcer"],
            "BPCER": b_frame["bpcer"],
            "ACER": b_frame["acer"],
            "AUC": b_frame.get("auc", 0.0)
        },
        {
            "Model": "ResNet34 + LBP (Fusion)",
            "Accuracy": f_frame["accuracy"],
            "Precision": f_frame["precision"],
            "Recall": f_frame["recall"],
            "F1-Score": f_frame["f1_score"],
            "APCER": f_frame["apcer"],
            "BPCER": f_frame["bpcer"],
            "ACER": f_frame["acer"],
            "AUC": f_frame.get("auc", 0.0)
        }
    ]

    comp_df = pd.DataFrame(comparison_rows)
    config = load_config(config_path)
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / config.get("paths", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)

    plot_path = results_dir / "model_comparison.png"
    plot_model_comparison(comp_df, str(plot_path))

    logger.info("\n" + "=" * 70)
    logger.info("  FINAL MODEL COMPARISON SUMMARY:")
    logger.info("=" * 70)
    print(comp_df.to_string(index=False))
    logger.info(f"Comparison chart saved to: {plot_path}")
    logger.info(f"Comparison CSV saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Anti-Spoofing Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_model.pth", help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "baseline"], help="Model architecture")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--compare", action="store_true", help="Run comparison between baseline and fusion models")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/best_baseline_model.pth", help="Path to baseline checkpoint")
    args = parser.parse_args()

    if args.compare:
        compare_models(baseline_ckpt=args.baseline_ckpt, fusion_ckpt=args.checkpoint, config_path=args.config)
    else:
        run_evaluation(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_type=args.model,
            split=args.split,
            threshold=args.threshold
        )
