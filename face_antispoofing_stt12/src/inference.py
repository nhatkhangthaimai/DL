"""
Face Anti-Spoofing Inference Engine
Performs real-time and offline prediction with temporal smoothing.
Author: Team STT 12
"""

from collections import deque
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model, ResNet34LBPFusion, ResNet34Baseline
from .texture import get_lbp_extractor, LBPFeatureExtractor
from .face_detection import FaceDetector, crop_face_with_margin
from .preprocessing import get_image_transforms
from .utils import load_config, get_device, load_checkpoint


class FaceAntiSpoofingPredictor:
    """
    High-performance inference engine for Face Anti-Spoofing.
    Supports both ResNet34 + LBP Fusion and ResNet34 Baseline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None,
        model_type: str = "fusion",
        smooth_window: int = 5
    ):
        self.config = config or load_config(config_path)
        self.device = torch.device(device) if device else get_device(self.config.get("project", {}).get("device", "cuda"))
        self.model_type = model_type
        self.threshold = self.config.get("inference", {}).get("threshold", 0.5)

        # 1. Initialize Feature Extractor & Preprocessor
        self.lbp_extractor = get_lbp_extractor(self.config)
        image_size = self.config.get("dataset", {}).get("image_size", 224)
        self.transform = get_image_transforms(image_size=image_size, is_train=False)

        # 2. Build and load model
        self.model = build_model(self.config, model_type=self.model_type)
        load_checkpoint(checkpoint_path, self.model, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()

        # 3. Temporal smoothing buffer for video stream
        self.smooth_window = smooth_window
        self.prob_history = deque(maxlen=smooth_window)

    @torch.no_grad()
    def predict_face(self, face_rgb_224: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a pre-cropped 224x224 RGB face image.
        Returns prediction dictionary with smoothed and raw probabilities.
        """
        # Prepare RGB Tensor
        rgb_tensor = self.transform(face_rgb_224).unsqueeze(0).to(self.device)

        if self.model_type == "baseline":
            logits = self.model(rgb_tensor)
        else:
            # Extract LBP texture features
            lbp_feat = self.lbp_extractor.extract_features(face_rgb_224)
            lbp_tensor = torch.tensor(lbp_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(rgb_tensor, lbp_tensor)

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        live_prob = float(probs[0])
        spoof_prob = float(probs[1])

        # Temporal smoothing
        self.prob_history.append(spoof_prob)
        smooth_spoof_prob = float(np.mean(self.prob_history))
        smooth_live_prob = 1.0 - smooth_spoof_prob

        # Decision based on threshold
        is_spoof = smooth_spoof_prob >= self.threshold
        label = 1 if is_spoof else 0
        label_name = "SPOOF" if is_spoof else "LIVE"
        confidence = smooth_spoof_prob if is_spoof else smooth_live_prob

        return {
            "label": label,
            "label_name": label_name,
            "confidence": confidence,
            "live_prob": smooth_live_prob,
            "spoof_prob": smooth_spoof_prob,
            "raw_live_prob": live_prob,
            "raw_spoof_prob": spoof_prob,
        }

    def reset_smoothing(self):
        """Clears temporal smoothing history (e.g., when face changes or disappears)."""
        self.prob_history.clear()
