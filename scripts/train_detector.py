#!/usr/bin/env python3
"""
Train License Plate Detection Model

Fine-tunes YOLOv8 on license plate detection dataset.
Supports GPU acceleration with RTX 5090/Blackwell (CUDA 12.8).
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
    else:
        print("No GPU available, using CPU (training will be slow)")
        return "cpu"


def train(
    data_config: str,
    model_size: str = "m",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    patience: int = 20,
    resume: bool = False,
    name: str = None
):
    """
    Train YOLOv8 license plate detector.

    Args:
        data_config: Path to dataset YAML config
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        img_size: Image size for training
        patience: Early stopping patience
        resume: Resume from last checkpoint
        name: Run name for logging
    """
    from ultralytics import YOLO

    # Validate paths
    data_config = Path(data_config)
    if not data_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")

    # Set up paths
    project_root = Path(__file__).parent.parent
    runs_dir = project_root / "ml" / "training" / "runs"
    models_dir = project_root / "ml" / "models"
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate run name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"license_plate_{model_size}_{timestamp}"

    print("=" * 60)
    print("License Plate Detection Training")
    print("=" * 60)
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Early stopping patience: {patience}")
    print(f"Run name: {name}")
    print("=" * 60)

    # Get device
    device = get_device()

    # Load pre-trained model
    base_model = f"yolov8{model_size}.pt"
    print(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)

    # Training arguments
    train_args = {
        "data": str(data_config.absolute()),
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "device": device,
        "patience": patience,
        "save": True,
        "save_period": 10,  # Save checkpoint every 10 epochs
        "project": str(runs_dir),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": True,  # Single class (license plate)
        "resume": resume,

        # Augmentation settings for license plates
        "augment": True,
        "degrees": 10.0,      # Rotation
        "translate": 0.1,     # Translation
        "scale": 0.5,         # Scale variation
        "shear": 2.0,         # Shear
        "perspective": 0.0,   # Perspective
        "flipud": 0.0,        # No vertical flip (plates don't flip)
        "fliplr": 0.5,        # Horizontal flip
        "mosaic": 1.0,        # Mosaic augmentation
        "mixup": 0.0,         # No mixup
        "copy_paste": 0.0,    # No copy-paste

        # Loss weights
        "box": 7.5,           # Box loss gain
        "cls": 0.5,           # Class loss gain
        "dfl": 1.5,           # DFL loss gain
    }

    # Start training
    print("\nStarting training...")
    print("-" * 60)

    try:
        results = model.train(**train_args)

        # Get best model path
        best_model = runs_dir / name / "weights" / "best.pt"
        last_model = runs_dir / name / "weights" / "last.pt"

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Print results
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\nFinal Metrics:")
            print(f"  mAP@0.5:      {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision:    {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall:       {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

        print(f"\nModel saved:")
        print(f"  Best:  {best_model}")
        print(f"  Last:  {last_model}")

        # Copy best model to models directory
        target_model = models_dir / "best.pt"
        if best_model.exists():
            import shutil
            shutil.copy2(best_model, target_model)
            print(f"\nBest model copied to: {target_model}")
            print("This model will be used by the application automatically.")

        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("You can resume training with --resume flag")
        return None

    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


def validate(model_path: str, data_config: str):
    """
    Validate a trained model.

    Args:
        model_path: Path to trained model weights
        data_config: Path to dataset YAML config
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("Model Validation")
    print("=" * 60)

    model = YOLO(model_path)
    device = get_device()

    results = model.val(
        data=data_config,
        device=device,
        split="test",
        verbose=True
    )

    print("\nValidation Results:")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 license plate detector"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--data",
        type=str,
        default="ml/training/configs/license_plate.yaml",
        help="Path to dataset config YAML"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    train_parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience"
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    train_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name for logging"
    )

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model")
    val_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    val_parser.add_argument(
        "--data",
        type=str,
        default="ml/training/configs/license_plate.yaml",
        help="Path to dataset config YAML"
    )

    args = parser.parse_args()

    # Default to train if no command specified
    if args.command is None:
        args.command = "train"
        args.data = "ml/training/configs/license_plate.yaml"
        args.model = "m"
        args.epochs = 100
        args.batch = 16
        args.img_size = 640
        args.patience = 20
        args.resume = False
        args.name = None

    project_root = Path(__file__).parent.parent

    if args.command == "train":
        data_config = project_root / args.data
        train(
            data_config=data_config,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            patience=args.patience,
            resume=args.resume,
            name=args.name
        )

    elif args.command == "validate":
        data_config = project_root / args.data
        validate(
            model_path=args.model,
            data_config=str(data_config)
        )


if __name__ == "__main__":
    main()
