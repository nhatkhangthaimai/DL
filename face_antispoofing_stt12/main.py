#!/usr/bin/env python3
"""
Main Entrypoint CLI for Face Anti-Spoofing Project (STT 12)
Author: Team STT 12
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Face Anti-Spoofing Detection for Authentication Systems (STT 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py check                   # Verify environment and audit raw dataset
  python main.py prepare --frames 12     # Preprocess MSU-MFSD videos into face crops
  python main.py train --model fusion    # Train ResNet34 + LBP fusion model
  python main.py train --model baseline  # Train ResNet34 baseline model
  python main.py evaluate --model fusion # Evaluate model on test set
  python main.py compare                 # Run comparison between Baseline and Fusion
  python main.py demo --camera 0         # Launch real-time webcam demo
  python main.py test                    # Run pytest suite
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. Check
    check_parser = subparsers.add_parser("check", help="Verify environment & audit dataset")

    # 2. Prepare
    prep_parser = subparsers.add_parser("prepare", help="Preprocess MSU-MFSD dataset")
    prep_parser.add_argument("--config", type=str, default="configs/config.yaml")
    prep_parser.add_argument("--frames", type=int, default=None, help="Uniform frames to sample per video")

    # 3. Train
    train_parser = subparsers.add_parser("train", help="Train deep learning model")
    train_parser.add_argument("--config", type=str, default="configs/config.yaml")
    train_parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "baseline"])
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--device", type=str, default=None)

    # 4. Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--config", type=str, default="configs/config.yaml")
    eval_parser.add_argument("--checkpoint", type=str, default=None)
    eval_parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "baseline"])
    eval_parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    eval_parser.add_argument("--threshold", type=float, default=0.5)

    # 5. Compare
    comp_parser = subparsers.add_parser("compare", help="Compare Baseline vs Fusion models")
    comp_parser.add_argument("--config", type=str, default="configs/config.yaml")
    comp_parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/best_baseline_model.pth")
    comp_parser.add_argument("--fusion-ckpt", type=str, default="checkpoints/best_fusion_model.pth")

    # 6. Demo
    demo_parser = subparsers.add_parser("demo", help="Run webcam live demo")
    demo_parser.add_argument("--config", type=str, default="configs/config.yaml")
    demo_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_model.pth")
    demo_parser.add_argument("--camera", type=int, default=0)
    demo_parser.add_argument("--model", type=str, default="fusion")

    # 7. Test
    test_parser = subparsers.add_parser("test", help="Run pytest unit & integration tests")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "check":
        from scripts.verify_environment import check_environment
        from scripts.check_dataset import audit_msu_mfsd
        print("\n>>> Step 1: Environment Check")
        check_environment()
        print("\n>>> Step 2: Dataset Audit")
        base_dir = Path(__file__).resolve().parent
        audit_msu_mfsd(base_dir.parent / "MSU-MFSD")

    elif args.command == "prepare":
        from prepare_dataset import prepare_dataset
        prepare_dataset(args.config, args.frames)

    elif args.command == "train":
        from train import train_model
        train_model(
            config_path=args.config,
            model_type=args.model,
            override_epochs=args.epochs,
            override_batch_size=args.batch_size,
            override_lr=args.lr,
            override_device=args.device
        )

    elif args.command == "evaluate":
        from evaluate import run_evaluation
        ckpt = args.checkpoint or f"checkpoints/best_{args.model}_model.pth"
        run_evaluation(
            checkpoint_path=ckpt,
            config_path=args.config,
            model_type=args.model,
            split=args.split,
            threshold=args.threshold
        )

    elif args.command == "compare":
        from evaluate import compare_models
        compare_models(
            baseline_ckpt=args.baseline_ckpt,
            fusion_ckpt=args.fusion_ckpt,
            config_path=args.config
        )

    elif args.command == "demo":
        from webcam import run_webcam
        run_webcam(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            camera_id=args.camera,
            model_type=args.model
        )

    elif args.command == "test":
        import pytest
        sys.exit(pytest.main(["-v", "tests"]))


if __name__ == "__main__":
    main()
