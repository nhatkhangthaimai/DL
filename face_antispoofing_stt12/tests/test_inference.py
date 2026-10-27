"""
Unit Tests for Inference and Smoothing
Author: Team STT 12
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.utils import save_checkpoint, load_config
from src.model import ResNet34LBPFusion
from src.inference import FaceAntiSpoofingPredictor


@pytest.fixture
def dummy_checkpoint(tmp_path):
    ckpt_path = tmp_path / "dummy_fusion.pth"
    model = ResNet34LBPFusion(num_classes=2, pretrained=False)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_acer": 0.05
    }
    torch.save(state, str(ckpt_path))
    return str(ckpt_path)


def test_inference_and_smoothing(dummy_checkpoint):
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(str(base_dir / "configs" / "config.yaml"))

    predictor = FaceAntiSpoofingPredictor(
        checkpoint_path=dummy_checkpoint,
        config=config,
        device="cpu",
        model_type="fusion",
        smooth_window=3
    )

    dummy_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = predictor.predict_face(dummy_rgb)

    assert "label" in res
    assert res["label"] in [0, 1]
    assert res["label_name"] in ["LIVE", "SPOOF"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["live_prob"] <= 1.0
    assert 0.0 <= res["spoof_prob"] <= 1.0
    assert np.isclose(res["live_prob"] + res["spoof_prob"], 1.0)
