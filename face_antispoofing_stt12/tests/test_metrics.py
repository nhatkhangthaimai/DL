"""
Unit Tests for Face Anti-Spoofing Metrics (APCER, BPCER, ACER, AUC)
Author: Team STT 12
"""

import pytest
import numpy as np
import pandas as pd

from src.metrics import compute_fas_metrics, find_optimal_threshold, compute_video_level_metrics


def test_perfect_predictions():
    # 0 = LIVE, 1 = SPOOF
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])

    m = compute_fas_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["apcer"] == 0.0  # Zero spoof errors
    assert m["bpcer"] == 0.0  # Zero live errors
    assert m["acer"] == 0.0   # Perfect ACER
    assert m["auc"] == 1.0


def test_all_spoofs_misclassified_as_live():
    # 3 Live correctly classified, 3 Spoofs incorrectly predicted as Live
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])

    m = compute_fas_metrics(y_true, y_pred)
    assert m["apcer"] == 1.0  # 100% of spoofs penetrated system as live
    assert m["bpcer"] == 0.0  # 0% of live rejected
    assert m["acer"] == 0.5   # (1.0 + 0.0) / 2


def test_all_live_misclassified_as_spoof():
    # 3 Live rejected as spoof, 3 Spoofs correctly identified as spoof
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1])

    m = compute_fas_metrics(y_true, y_pred)
    assert m["apcer"] == 0.0  # 0% of spoofs penetrated
    assert m["bpcer"] == 1.0  # 100% of bona fide live rejected
    assert m["acer"] == 0.5


def test_video_level_aggregation():
    # 2 videos: vid1 (Live), vid2 (Spoof)
    # vid1 has 3 frames: [0.1, 0.2, 0.3] -> mean = 0.2 < 0.5 -> Predicted LIVE (Correct)
    # vid2 has 3 frames: [0.8, 0.9, 0.7] -> mean = 0.8 >= 0.5 -> Predicted SPOOF (Correct)
    dummy_df = pd.DataFrame({
        "video_name": ["vid1", "vid1", "vid1", "vid2", "vid2", "vid2"],
        "label": [0, 0, 0, 1, 1, 1],
        "pred_prob": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]
    })

    v_metrics = compute_video_level_metrics(dummy_df, threshold=0.5)
    assert v_metrics["total_videos"] == 2
    assert v_metrics["accuracy"] == 1.0
    assert v_metrics["acer"] == 0.0
