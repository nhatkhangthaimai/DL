#!/usr/bin/env python3
"""
Real-time Face Anti-Spoofing Webcam Demo for Access Control & Attendance Systems
Author: Team STT 12 - Face Anti-Spoofing
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.inference import FaceAntiSpoofingPredictor
from src.face_detection import FaceDetector, crop_face_with_margin
from src.utils import load_config, setup_logger, FPSCounter


def run_webcam(
    checkpoint_path: str = "checkpoints/best_fusion_model.pth",
    config_path: str = "configs/config.yaml",
    camera_id: int = 0,
    model_type: str = "fusion"
):
    logger = setup_logger("WebcamDemo", "logs/webcam.log")
    logger.info("=" * 65)
    logger.info("   LAUNCHING REALTIME FACE ANTI-SPOOFING WEBCAM DEMO")
    logger.info(f"   Model: {model_type.upper()} | Checkpoint: {checkpoint_path}")
    logger.info("=" * 65)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        logger.error(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            f"Please train the model first by running:\n"
            f"  python train.py --model {model_type}\n"
            f"Or specify an existing checkpoint with --checkpoint <path>"
        )
        sys.exit(1)

    config = load_config(config_path)

    # Initialize Face Detector and FAS Predictor
    detector = FaceDetector(min_size=(70, 70))
    predictor = FaceAntiSpoofingPredictor(
        checkpoint_path=str(ckpt_file),
        config=config,
        model_type=model_type,
        smooth_window=config.get("inference", {}).get("smooth_window", 5)
    )

    # Open Camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_id}. Check if webcam is connected or in use by another app.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = FPSCounter()
    window_name = "Face Anti-Spoofing System [STT 12] - Attendance Authentication Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1020, 680)

    logger.info("[+] Camera initialized. Press 'Q' or 'ESC' in the demo window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from webcam.")
            break

        current_fps = fps_counter.tick()
        h, w = frame.shape[:2]

        # 1. Detect faces
        faces = detector.detect_faces(frame)

        if not faces:
            predictor.reset_smoothing()
            # Status banner: No face
            cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.putText(
                frame,
                "STATUS: [NO FACE DETECTED] - Please align face within view",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (180, 180, 180),
                2,
                cv2.LINE_AA
            )
        else:
            # Process each detected face (primary face = largest)
            for idx, (left, top, right, bottom, _) in enumerate(faces):
                cropped_face_bgr, expanded_box = crop_face_with_margin(
                    frame, (left, top, right, bottom), margin_ratio=0.15, target_size=(224, 224)
                )

                if cropped_face_bgr is None:
                    continue

                # Convert to RGB for inference
                cropped_face_rgb = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB)

                # 2. FAS Inference
                res = predictor.predict_face(cropped_face_rgb)
                is_spoof = (res["label"] == 1)
                label_text = res["label_name"]
                conf_pct = res["confidence"] * 100.0

                # 3. Styling: Live = Neon Green, Spoof = Crimson Red
                if is_spoof:
                    box_color = (30, 30, 230)  # Red (BGR)
                    status_text = "[ACCESS DENIED] Spoof Attack Detected!"
                    status_color = (30, 30, 240)
                else:
                    box_color = (40, 210, 40)  # Green (BGR)
                    status_text = "[ACCESS GRANTED] Genuine Live Face Verified"
                    status_color = (40, 220, 40)

                ex1, ey1, ex2, ey2 = expanded_box

                # Draw bounding box with corner accents
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), box_color, 2)
                corner_len = int((ex2 - ex1) * 0.15)
                # Top-left corner
                cv2.line(frame, (ex1, ey1), (ex1 + corner_len, ey1), box_color, 4)
                cv2.line(frame, (ex1, ey1), (ex1, ey1 + corner_len), box_color, 4)
                # Top-right corner
                cv2.line(frame, (ex2, ey1), (ex2 - corner_len, ey1), box_color, 4)
                cv2.line(frame, (ex2, ey1), (ex2, ey1 + corner_len), box_color, 4)
                # Bottom-left corner
                cv2.line(frame, (ex1, ey2), (ex1 + corner_len, ey2), box_color, 4)
                cv2.line(frame, (ex1, ey2), (ex1, ey2 - corner_len), box_color, 4)
                # Bottom-right corner
                cv2.line(frame, (ex2, ey2), (ex2 - corner_len, ey2), box_color, 4)
                cv2.line(frame, (ex2, ey2), (ex2, ey2 - corner_len), box_color, 4)

                # Label tag above box
                tag_str = f"{label_text}: {conf_pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tag_y = max(ey1 - 10, th + 10)
                cv2.rectangle(frame, (ex1, tag_y - th - 8), (ex1 + tw + 14, tag_y + 4), box_color, -1)
                cv2.putText(
                    frame,
                    tag_str,
                    (ex1 + 7, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Attendance Gate decision banner (for primary face)
                if idx == 0:
                    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
                    cv2.putText(
                        frame,
                        f"STATUS: {status_text} (Live: {res['live_prob']*100:.1f}%, Spoof: {res['spoof_prob']*100:.1f}%)",
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        status_color,
                        2,
                        cv2.LINE_AA
                    )

        # Bottom HUD: FPS, Architecture, Controls
        cv2.rectangle(frame, (0, h - 45), (w, h), (15, 15, 15), -1)
        hud_text = f"FPS: {current_fps:.1f} | Arch: ResNet34 + LBP Texture Fusion | STT 12 | Press 'Q' to Exit"
        cv2.putText(
            frame,
            hud_text,
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("[+] Webcam demo terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Face Anti-Spoofing Webcam Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_model.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "baseline"], help="Model type")
    args = parser.parse_args()

    run_webcam(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        camera_id=args.camera,
        model_type=args.model
    )
