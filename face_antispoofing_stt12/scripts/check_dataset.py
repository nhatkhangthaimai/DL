#!/usr/bin/env python3
"""
MSU-MFSD Dataset Audit & Integrity Check
Author: Team STT 12 - Face Anti-Spoofing
"""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict


def parse_video_filename(filename: str):
    """
    Parses MSU-MFSD naming format:
    real: real_client001_android_SD_scene01.mp4 / .mov
    attack: attack_client001_android_SD_ipad_video_scene01.mp4 / .mov
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()

    # Pattern for real
    # real_client<ID>_<device>_<resolution>_scene01
    real_match = re.match(r"^real_client(\d+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(scene01)$", stem)
    if real_match:
        return {
            "type": "real",
            "client_id": real_match.group(1),
            "device": real_match.group(2),
            "resolution": real_match.group(3),
            "attack_type": "none",
            "scenario": real_match.group(4),
            "extension": ext
        }

    # Pattern for attack
    # attack_client<ID>_<device>_<resolution>_<attackType>_scene01
    attack_match = re.match(r"^attack_client(\d+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(ipad_video|iphone_video|printed_photo)_(scene01)$", stem)
    if attack_match:
        return {
            "type": "attack",
            "client_id": attack_match.group(1),
            "device": attack_match.group(2),
            "resolution": attack_match.group(3),
            "attack_type": attack_match.group(4),
            "scenario": attack_match.group(5),
            "extension": ext
        }

    return None


def audit_msu_mfsd(dataset_path: Path):
    print("=" * 70)
    print("           MSU-MFSD DATASET AUDIT AND VERIFICATION")
    print("=" * 70)

    if not dataset_path.exists():
        print(f"[-] ERROR: Dataset path {dataset_path} does not exist!")
        return False

    real_dir = dataset_path / "scene01" / "real"
    attack_dir = dataset_path / "scene01" / "attack"
    train_sub_file = dataset_path / "train_sub_list.txt"
    test_sub_file = dataset_path / "test_sub_list.txt"

    # 1. Read Train and Test subject lists
    train_subjects = set()
    if train_sub_file.exists():
        with open(train_sub_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    train_subjects.add(s.zfill(3))

    test_subjects = set()
    if test_sub_file.exists():
        with open(test_sub_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    test_subjects.add(s.zfill(3))

    print(f"\n[1] Official Protocol Split Lists:")
    print(f"    - Train Subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    print(f"    - Test Subjects  ({len(test_subjects)}): {sorted(list(test_subjects))}")

    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        print(f"    [!] CRITICAL WARNING: Overlap found between Train and Test: {overlap}")
    else:
        print(f"    [+] Zero Subject Leakage: Train and Test subject sets are completely disjoint.")

    # 2. Inspect Real videos and .face files
    real_video_files = []
    real_face_files = set()
    if real_dir.exists():
        for p in real_dir.iterdir():
            if p.suffix.lower() in [".mp4", ".mov"]:
                real_video_files.append(p)
            elif p.suffix.lower() == ".face":
                real_face_files.add(p.stem)

    # 3. Inspect Attack videos and .face files
    attack_video_files = []
    attack_face_files = set()
    if attack_dir.exists():
        for p in attack_dir.iterdir():
            if p.suffix.lower() in [".mp4", ".mov"]:
                attack_video_files.append(p)
            elif p.suffix.lower() == ".face":
                attack_face_files.add(p.stem)

    print(f"\n[2] Video Files Count:")
    print(f"    - Real Videos:   {len(real_video_files)} (Expected: 70)")
    print(f"    - Attack Videos: {len(attack_video_files)} (Expected: 210)")
    print(f"    - Total Videos:  {len(real_video_files) + len(attack_video_files)} (Expected: 280)")

    # 4. Check .face bounding box files matching
    real_missing_faces = [v.name for v in real_video_files if v.stem not in real_face_files]
    attack_missing_faces = [v.name for v in attack_video_files if v.stem not in attack_face_files]

    print(f"\n[3] Face Annotation (.face) File Coverage:")
    print(f"    - Real .face files:   {len(real_face_files)} / {len(real_video_files)}")
    print(f"    - Attack .face files: {len(attack_face_files)} / {len(attack_video_files)}")
    if real_missing_faces:
        print(f"    [!] Real videos missing .face: {real_missing_faces}")
    if attack_missing_faces:
        print(f"    [!] Attack videos missing .face: {attack_missing_faces}")
    if not real_missing_faces and not attack_missing_faces:
        print(f"    [+] 100% of videos have corresponding .face bounding box annotations!")

    # 5. Parse and aggregate by Subject ID, Device, Attack type
    all_videos = real_video_files + attack_video_files
    parsed_info = []
    unparsed_videos = []
    subject_videos = defaultdict(lambda: {"real": 0, "attack": 0, "attacks_by_type": defaultdict(int), "devices": defaultdict(int)})

    for v in all_videos:
        meta = parse_video_filename(v.name)
        if meta:
            meta["path"] = str(v)
            parsed_info.append(meta)
            sub_id = meta["client_id"]
            if meta["type"] == "real":
                subject_videos[sub_id]["real"] += 1
            else:
                subject_videos[sub_id]["attack"] += 1
                subject_videos[sub_id]["attacks_by_type"][meta["attack_type"]] += 1
            subject_videos[sub_id]["devices"][meta["device"]] += 1
        else:
            unparsed_videos.append(v.name)

    if unparsed_videos:
        print(f"\n[!] Warning: Some videos did not match expected naming convention:")
        for uv in unparsed_videos:
            print(f"    - {uv}")

    print(f"\n[4] Subject Representation:")
    total_subjects = len(subject_videos)
    print(f"    - Total unique subjects found in files: {total_subjects} (Expected: 35)")

    train_vids_real = 0
    train_vids_attack = 0
    test_vids_real = 0
    test_vids_attack = 0
    unassigned_vids = 0

    for item in parsed_info:
        cid = item["client_id"]
        if cid in train_subjects:
            if item["type"] == "real":
                train_vids_real += 1
            else:
                train_vids_attack += 1
        elif cid in test_subjects:
            if item["type"] == "real":
                test_vids_real += 1
            else:
                test_vids_attack += 1
        else:
            unassigned_vids += 1

    print(f"\n[5] Split Protocol Conformity:")
    print(f"    - Official Train Split: {train_vids_real} Real, {train_vids_attack} Attack (Total: {train_vids_real + train_vids_attack}) [Expected: 30 Real, 90 Attack = 120]")
    print(f"    - Official Test Split:  {test_vids_real} Real, {test_vids_attack} Attack (Total: {test_vids_real + test_vids_attack}) [Expected: 40 Real, 120 Attack = 160]")
    if unassigned_vids > 0:
        print(f"    [!] Videos belonging to subjects not in train or test: {unassigned_vids}")
    else:
        print(f"    [+] All 280 videos strictly belong to either official Train or Test subject sets.")

    # 6. Attack Type Breakdown
    attack_types_count = defaultdict(int)
    device_count = defaultdict(int)
    for item in parsed_info:
        if item["type"] == "attack":
            attack_types_count[item["attack_type"]] += 1
        device_count[item["device"]] += 1

    print(f"\n[6] Attack Types Distribution:")
    for atype, cnt in attack_types_count.items():
        print(f"    - {atype:<15}: {cnt} videos (Expected: 70 each)")

    print(f"\n[7] Camera/Device Distribution:")
    for dev, cnt in device_count.items():
        print(f"    - {dev:<15}: {cnt} videos (Expected: 140 each)")

    # 7. Check .face content format
    sample_face = real_dir / "real_client001_android_SD_scene01.face"
    if sample_face.exists():
        with open(sample_face, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n[8] Sample .face File Inspection:")
        print(f"    - Sample file: {sample_face.name}")
        print(f"    - Annotated frames count: {len(lines)}")
        if len(lines) > 0:
            print(f"    - Line 0 format: {lines[0]}")

    print("\n" + "=" * 70)
    print("                    AUDIT COMPLETED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent.parent / "MSU-MFSD"
    audit_msu_mfsd(dataset_dir)
