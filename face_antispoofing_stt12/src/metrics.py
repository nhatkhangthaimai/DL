"""
Face Anti-Spoofing Performance Evaluation Metrics
Includes standard classification metrics and ISO/IEC 30107-3 PAD metrics (APCER, BPCER, ACER)
Author: Team STT 12
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def compute_fas_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard metrics and Face Anti-Spoofing (PAD) metrics.
    Conventions:
      0 = LIVE (Bona Fide)
      1 = SPOOF (Presentation Attack)

    APCER (Attack Presentation Classification Error Rate):
      Ratio of Spoof samples erroneously classified as Live.
      APCER = Count(True == 1 and Pred == 0) / Count(True == 1)

    BPCER (Bona Fide Presentation Classification Error Rate):
      Ratio of Live samples erroneously classified as Spoof.
      BPCER = Count(True == 0 and Pred == 1) / Count(True == 0)

    ACER (Average Classification Error Rate):
      ACER = (APCER + BPCER) / 2
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm: [[TN (True Live), FP (Live as Spoof)],
    #      [FN (Spoof as Live), TP (True Spoof)]]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    num_live = tn + fp
    num_spoof = fn + tp

    # APCER: Spoof misclassified as Live / Total Spoofs
    apcer = (fn / num_spoof) if num_spoof > 0 else 0.0

    # BPCER: Live misclassified as Spoof / Total Live
    bpcer = (fp / num_live) if num_live > 0 else 0.0

    # ACER: Average of APCER and BPCER
    acer = (apcer + bpcer) / 2.0

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist()
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        results["auc"] = float(roc_auc)
        results["fpr"] = fpr.tolist()
        results["tpr"] = tpr.tolist()
        results["thresholds"] = thresholds.tolist()

        # Compute Equal Error Rate (EER) where FPR == 1 - TPR (or APCER == BPCER)
        # Note: In ISO PAD, BPCER is FPR and APCER is FNR (1 - TPR)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        results["eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        results["eer_threshold"] = float(thresholds[eer_idx])

    return results


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Finds decision threshold on validation set that minimizes ACER.
    Returns: (optimal_threshold, min_acer)
    """
    threshold_candidates = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    min_acer = float("inf")

    for th in threshold_candidates:
        preds = (y_prob >= th).astype(int)
        m = compute_fas_metrics(y_true, preds)
        if m["acer"] < min_acer:
            min_acer = m["acer"]
            best_thresh = float(th)

    return best_thresh, min_acer


def compute_video_level_metrics(
    eval_df: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Aggregates frame-level predictions to video-level predictions by averaging
    probabilities for frames of the same video.
    eval_df columns required: 'video_name', 'label', 'pred_prob'
    """
    grouped = eval_df.groupby("video_name").agg({
        "label": "first",
        "pred_prob": "mean"
    }).reset_index()

    y_true = grouped["label"].values
    y_prob = grouped["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_fas_metrics(y_true, y_pred, y_prob)
    metrics["total_videos"] = len(grouped)
    return metrics


# ==============================================================================
# Plotting and Reporting Utilities
# ==============================================================================

def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    """Saves formatted confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    labels = ["LIVE (0)", "SPOOF (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth Label", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float, save_path: str, eer: Optional[float] = None):
    """Saves ROC curve with AUC and EER annotations."""
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#0066ff", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#707070", linestyle="--", lw=1.5, label="Random Chance")

    if eer is not None:
        plt.plot([eer], [1 - eer], marker="o", markersize=8, color="#e63946", label=f"EER = {eer:.4f}")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (BPCER)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - APCER)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """Plots training and validation loss and accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", color="#0066ff", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], "s-", color="#e63946", label="Val Loss", lw=2)
    ax1.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy / ACER subplot
    if "train_acc" in history and "val_acc" in history:
        ax2.plot(epochs, history["train_acc"], "o-", color="#2a9d8f", label="Train Acc", lw=2)
        ax2.plot(epochs, history["val_acc"], "s-", color="#e76f51", label="Val Acc", lw=2)
        ax2.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str):
    """
    Plots side-by-side bar chart comparing ResNet34 Baseline vs ResNet34 + LBP Fusion.
    """
    plt.figure(figsize=(9, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ACER"]

    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    plot_df = comparison_df.set_index("Model")[available_metrics].T

    plot_df.plot(kind="bar", figsize=(10, 5), width=0.7, colormap="tab10")
    plt.title("Model Comparison: ResNet34 Baseline vs ResNet34 + LBP Fusion", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score / Rate", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
