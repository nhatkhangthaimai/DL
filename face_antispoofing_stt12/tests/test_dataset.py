"""
Unit Tests for Dataset Preprocessing and Loading
Author: Team STT 12
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.utils import load_config
from src.texture import LBPFeatureExtractor
from src.dataset import FaceAntiSpoofingDataset, compute_class_weights
from scripts.check_dataset import parse_video_filename


@pytest.fixture
def sample_config():
    base_dir = Path(__file__).resolve().parent.parent
    return load_config(str(base_dir / "configs" / "config.yaml"))


def test_filename_parsing():
    # Real video filename
    real_meta = parse_video_filename("real_client001_android_SD_scene01.mp4")
    assert real_meta is not None
    assert real_meta["type"] == "real"
    assert real_meta["client_id"] == "001"
    assert real_meta["device"] == "android"

    # Attack video filename
    attack_meta = parse_video_filename("attack_client005_laptop_SD_ipad_video_scene01.mov")
    assert attack_meta is not None
    assert attack_meta["type"] == "attack"
    assert attack_meta["client_id"] == "005"
    assert attack_meta["attack_type"] == "ipad_video"
    assert attack_meta["device"] == "laptop"


def test_zero_data_leakage(sample_config):
    base_dir = Path(__file__).resolve().parent.parent
    metadata_csv = base_dir / "processed" / "metadata.csv"
    if not metadata_csv.exists():
        pytest.skip("metadata.csv not yet generated, skipping data leakage check.")

    df = pd.read_csv(metadata_csv)
    train_subs = set(df[df["split"] == "train"]["subject_id"].unique())
    val_subs = set(df[df["split"] == "val"]["subject_id"].unique())
    test_subs = set(df[df["split"] == "test"]["subject_id"].unique())

    # Verify sets are completely disjoint
    assert len(train_subs.intersection(val_subs)) == 0, f"Leakage: train & val overlap on {train_subs.intersection(val_subs)}"
    assert len(train_subs.intersection(test_subs)) == 0, f"Leakage: train & test overlap on {train_subs.intersection(test_subs)}"
    assert len(val_subs.intersection(test_subs)) == 0, f"Leakage: val & test overlap on {val_subs.intersection(test_subs)}"


def test_lbp_feature_extraction():
    extractor = LBPFeatureExtractor(radius=1, n_points=8, method="uniform", grid_x=4, grid_y=4)
    # Synthetic 224x224 RGB image
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    feat = extractor.extract_features(dummy_img)
    assert feat.ndim == 1
    assert len(feat) == extractor.feature_dim
    assert np.isfinite(feat).all()
    # Norm should be approx 1.0 (L2 normalized)
    assert np.isclose(np.linalg.norm(feat), 1.0, atol=1e-3)


def test_class_weights_computation():
    dummy_df = pd.DataFrame({
        "label": [0, 0, 1, 1, 1, 1]  # 2 Live, 4 Spoof
    })
    weights = compute_class_weights(dummy_df)
    assert len(weights) == 2
    # Live weight should be higher because it is the minority class
    assert weights[0] > weights[1]
    assert torch.isclose(weights[0], torch.tensor(1.5))
    assert torch.isclose(weights[1], torch.tensor(0.75))
