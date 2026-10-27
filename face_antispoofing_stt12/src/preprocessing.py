"""
Video and Face Preprocessing Pipeline
Author: Team STT 12 - Face Anti-Spoofing
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from .face_detection import FaceAnnotationReader, FaceDetector, crop_face_with_margin


def get_image_transforms(image_size: int = 224, is_train: bool = True):
    """
    Returns standard ImageNet normalization transforms for ResNet34.
    Augmentation is kept moderate so as not to destroy high-frequency LBP texture cues.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def check_video_rotation(video_path: Path) -> int:
    """
    Checks if video has 180-degree rotation metadata (common in Android front cameras).
    Returns rotation angle (0 or 180).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    # In modern OpenCV, cv2.CAP_PROP_ORIENTATION_META retrieves EXIF rotation
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        cap.release()
        if rotation == 180:
            return 180
    except Exception:
        cap.release()

    # Heuristic based on filename: MSU-MFSD android videos with face coordinates inverted
    return 0


def extract_uniform_frames(
    video_path: Path,
    num_frames: int = 12
) -> List[Tuple[int, np.ndarray]]:
    """
    Extracts num_frames uniformly distributed across the video timeline.
    returns: List of (frame_index, frame_bgr_image)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback reading sequentially if frame count not reported in container header
        frames_list = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append((idx, frame))
            idx += 1
        cap.release()
        if not frames_list:
            return []
        step = max(1, len(frames_list) // num_frames)
        return [frames_list[i] for i in range(0, len(frames_list), step)][:num_frames]

    # Calculate uniform indices across 5% to 95% of video to avoid transition artifacts
    start_f = int(total_frames * 0.05)
    end_f = int(total_frames * 0.95)
    if end_f <= start_f:
        target_indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    else:
        target_indices = np.linspace(start_f, end_f, num=min(num_frames, total_frames), dtype=int)

    extracted = []
    for f_idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            extracted.append((int(f_idx), frame))

    cap.release()
    return extracted


def process_video_frames(
    video_path: Path,
    face_annotation_path: Optional[Path] = None,
    num_frames: int = 12,
    margin_ratio: float = 0.15,
    target_size: Tuple[int, int] = (224, 224),
    fallback_detector: Optional[FaceDetector] = None
) -> List[Tuple[int, np.ndarray, Tuple[int, int, int, int]]]:
    """
    Reads uniform frames from video, crops the face region using .face annotation
    (or fallback detector), and resizes to target_size.
    returns: List of (frame_index, cropped_rgb_face_image, bbox)
    """
    anno_reader = FaceAnnotationReader(str(face_annotation_path)) if face_annotation_path and face_annotation_path.exists() else None
    raw_frames = extract_uniform_frames(video_path, num_frames=num_frames)

    processed_faces = []
    for frame_idx, frame in raw_frames:
        bbox = None
        if anno_reader is not None:
            bbox = anno_reader.get_bbox(frame_idx)

        # Fallback to Haar detector if .face annotation is missing
        if bbox is None and fallback_detector is not None:
            detected = fallback_detector.detect_faces(frame)
            if detected:
                l, t, r, b, _ = detected[0]
                bbox = (l, t, r, b)

        # If still no face detected, crop center square as graceful fallback
        if bbox is None:
            h, w = frame.shape[:2]
            sz = min(h, w)
            cy, cx = h // 2, w // 2
            bbox = (cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2)

        cropped_bgr, final_bbox = crop_face_with_margin(
            frame, bbox, margin_ratio=margin_ratio, target_size=target_size
        )

        if cropped_bgr is not None:
            # Convert BGR to RGB
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            processed_faces.append((frame_idx, cropped_rgb, final_bbox))

    return processed_faces
