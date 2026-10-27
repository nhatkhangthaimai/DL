#!/usr/bin/env python3
"""
Verify Environment and Dependencies
Author: Team STT 12 - Face Anti-Spoofing
"""

import sys
import platform
import os
from pathlib import Path


def check_environment():
    print("=" * 60)
    print("   FACE ANTI-SPOOFING SYSTEM: ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. Python info
    print(f"[+] Python Version: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"[+] Executable: {sys.executable}")
    print(f"[+] OS: {platform.system()} {platform.release()} ({platform.version()})")

    # 2. Check essential packages
    modules_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("skimage", "Scikit-Image"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("pytest", "Pytest"),
    ]

    all_ok = True
    print("\n--- Package Verification ---")
    for mod_name, disp_name in modules_to_check:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {disp_name:<16}: v{ver}")
        except ImportError:
            print(f"  [MISSING] {disp_name:<16}: Not installed!")
            all_ok = False

    # 3. Check PyTorch & CUDA
    try:
        import torch
        print("\n--- Deep Learning Hardware ---")
        cuda_avail = torch.cuda.is_available()
        print(f"  [+] CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            vram_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            print(f"  [+] GPU Device #{current_device}: {gpu_name}")
            print(f"  [+] Total VRAM: {vram_gb:.2f} GB")
            print(f"  [+] PyTorch CUDA Version: {torch.version.cuda}")
        else:
            print("  [-] Running on CPU mode.")
    except ImportError:
        pass

    # 4. Check Dataset presence
    print("\n--- Dataset Paths Verification ---")
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    workspace_dir = project_dir.parent

    dataset_dir = workspace_dir / "MSU-MFSD"
    print(f"  [+] Expected Dataset Directory: {dataset_dir}")
    if dataset_dir.exists():
        real_dir = dataset_dir / "scene01" / "real"
        attack_dir = dataset_dir / "scene01" / "attack"
        train_list = dataset_dir / "train_sub_list.txt"
        test_list = dataset_dir / "test_sub_list.txt"

        print(f"      - MSU-MFSD found: Yes")
        print(f"      - scene01/real exists: {real_dir.exists()}")
        print(f"      - scene01/attack exists: {attack_dir.exists()}")
        print(f"      - train_sub_list.txt exists: {train_list.exists()}")
        print(f"      - test_sub_list.txt exists: {test_list.exists()}")
    else:
        print(f"      [-] MSU-MFSD NOT found at {dataset_dir}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("   ALL CRITICAL PREREQUISITES VERIFIED SUCCESSFULLY!")
    else:
        print("   SOME PACKAGES OR PATHS REQUIRE ATTENTION.")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    check_environment()
