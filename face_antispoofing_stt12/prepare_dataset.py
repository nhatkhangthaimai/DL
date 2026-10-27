#!/usr/bin/env python3
"""
Automated Dataset Preprocessing Pipeline for MSU-MFSD
Author: Team STT 12 - Face Anti-Spoofing
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import cv2
from tqdm import tqdm
from PIL import Image

from src.utils import load_config, setup_logger
from src.preprocessing import process_video_frames
from src.face_detection import FaceDetector
from scripts.check_dataset import parse_video_filename


def prepare_dataset(config_path: str = "configs/config.yaml", sample_frames: int = None):
    config = load_config(config_path)
    logger = setup_logger("PrepareDataset", "logs/prepare_dataset.log")

    logger.info("=" * 60)
    logger.info("  STARTING MSU-MFSD DATASET PREPROCESSING")
    logger.info("=" * 60)

    # Resolve paths
    base_dir = Path(__file__).resolve().parent
    raw_dir_cfg = config.get("paths", {}).get("raw_dataset_dir", "../MSU-MFSD")
    raw_dataset_dir = (base_dir / raw_dir_cfg).resolve()
    processed_dir = (base_dir / config.get("paths", {}).get("processed_dir", "processed")).resolve()

    logger.info(f"Raw dataset path: {raw_dataset_dir}")
    logger.info(f"Target processed path: {processed_dir}")

    if not raw_dataset_dir.exists():
        logger.error(f"Raw dataset directory does not exist: {raw_dataset_dir}")
        sys.exit(1)

    # Read official protocol splits
    train_sub_file = raw_dataset_dir / "train_sub_list.txt"
    test_sub_file = raw_dataset_dir / "test_sub_list.txt"

    train_subjects_official = set()
    with open(train_sub_file, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                train_subjects_official.add(s.zfill(3))

    test_subjects = set()
    with open(test_sub_file, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                test_subjects.add(s.zfill(3))

    # Carve validation subjects out of train subjects (strictly subject-disjoint)
    val_sub_cfg = config.get("dataset", {}).get("val_subjects", ["34", "53", "54"])
    val_subjects = set(s.zfill(3) for s in val_sub_cfg)
    train_subjects = train_subjects_official - val_subjects

    logger.info(f"Subject Partitioning:")
    logger.info(f"  - Train Subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    logger.info(f"  - Val Subjects   ({len(val_subjects)}): {sorted(list(val_subjects))}")
    logger.info(f"  - Test Subjects  ({len(test_subjects)}): {sorted(list(test_subjects))}")

    # Ensure zero data leakage
    assert len(train_subjects.intersection(val_subjects)) == 0, "Leakage between train and val!"
    assert len(train_subjects.intersection(test_subjects)) == 0, "Leakage between train and test!"
    assert len(val_subjects.intersection(test_subjects)) == 0, "Leakage between val and test!"
    logger.info("[+] Verified 100% Subject-Disjoint splits (Zero Data Leakage).")

    # Destination directories
    splits = ["train", "val", "test"]
    categories = ["live", "spoof"]
    for s in splits:
        for c in categories:
            (processed_dir / s / c).mkdir(parents=True, exist_ok=True)

    # Collect videos
    real_vids = list((raw_dataset_dir / "scene01" / "real").glob("*.mp4")) + list((raw_dataset_dir / "scene01" / "real").glob("*.mov"))
    attack_vids = list((raw_dataset_dir / "scene01" / "attack").glob("*.mp4")) + list((raw_dataset_dir / "scene01" / "attack").glob("*.mov"))
    all_videos = real_vids + attack_vids
    logger.info(f"Found {len(all_videos)} total videos to process ({len(real_vids)} real, {len(attack_vids)} attack).")

    num_frames = sample_frames or config.get("dataset", {}).get("frames_per_video", 12)
    margin_ratio = config.get("dataset", {}).get("margin_ratio", 0.15)
    image_size = config.get("dataset", {}).get("image_size", 224)

    fallback_detector = FaceDetector()
    metadata_records = []
    success_videos = 0
    failed_videos = 0

    pbar = tqdm(all_videos, desc="Preprocessing Videos")
    for v_path in pbar:
        v_meta = parse_video_filename(v_path.name)
        if not v_meta:
            logger.warning(f"Skipping unparseable video filename: {v_path.name}")
            failed_videos += 1
            continue

        cid = v_meta["client_id"]
        v_type = v_meta["type"]  # "real" or "attack"

        # Determine split
        if cid in train_subjects:
            split_name = "train"
        elif cid in val_subjects:
            split_name = "val"
        elif cid in test_subjects:
            split_name = "test"
        else:
            logger.warning(f"Client ID {cid} not found in train or test protocol list!")
            continue

        label = 0 if v_type == "real" else 1
        label_name = "live" if label == 0 else "spoof"

        # Locate .face file in same directory
        face_path = v_path.with_suffix(".face")

        # Process frames
        processed_faces = process_video_frames(
            video_path=v_path,
            face_annotation_path=face_path,
            num_frames=num_frames,
            margin_ratio=margin_ratio,
            target_size=(image_size, image_size),
            fallback_detector=fallback_detector
        )

        if not processed_faces:
            logger.warning(f"Could not extract any valid face frames from {v_path.name}")
            failed_videos += 1
            continue

        success_videos += 1

        # Save extracted face images
        for f_idx, rgb_img, bbox in processed_faces:
            img_filename = f"{v_path.stem}_f{f_idx:04d}.jpg"
            img_dest = processed_dir / split_name / label_name / img_filename

            # Save as RGB JPEG
            Image.fromarray(rgb_img).save(img_dest, quality=95)

            # Record metadata
            metadata_records.append({
                "image_path": str(img_dest.relative_to(base_dir)).replace("\\", "/"),
                "absolute_path": str(img_dest).replace("\\", "/"),
                "video_name": v_path.stem,
                "video_file": v_path.name,
                "subject_id": cid,
                "label": label,
                "label_name": label_name.upper(),
                "split": split_name,
                "device": v_meta["device"],
                "attack_type": v_meta["attack_type"],
                "resolution": v_meta["resolution"],
                "frame_idx": f_idx,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
            })

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_records)
    csv_path = processed_dir / "metadata.csv"
    meta_df.to_csv(csv_path, index=False)
    logger.info(f"Metadata saved to: {csv_path}")

    # Log summary statistics
    logger.info("=" * 60)
    logger.info("  PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully processed videos: {success_videos} / {len(all_videos)}")
    logger.info(f"Failed videos: {failed_videos}")
    logger.info(f"Total extracted face images: {len(meta_df)}")

    for s in splits:
        s_df = meta_df[meta_df["split"] == s]
        s_live = len(s_df[s_df["label"] == 0])
        s_spoof = len(s_df[s_df["label"] == 1])
        s_subjects = s_df["subject_id"].nunique()
        logger.info(f"  Split '{s:<5}': {len(s_df):<5} images ({s_live} Live, {s_spoof} Spoof) across {s_subjects} subjects")

    logger.info("=" * 60)
    logger.info("[+] Dataset preprocessing completed successfully!")
    return meta_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MSU-MFSD dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to extract per video")
    args = parser.parse_args()

    prepare_dataset(args.config, args.frames)
