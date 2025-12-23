#!/usr/bin/env python3
"""
Evaluate License Plate Detection Model

Calculates metrics, generates visualizations, and tests on sample images.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_model(
    model_path: str,
    data_config: str,
    split: str = "test",
    save_predictions: bool = True,
    conf_threshold: float = 0.5
) -> Dict:
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to trained model weights
        data_config: Path to dataset config YAML
        split: Data split to evaluate on
        save_predictions: Whether to save prediction images
        conf_threshold: Confidence threshold for predictions

    Returns:
        Evaluation metrics dictionary
    """
    import torch
    from ultralytics import YOLO

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Model: {model_path}")
    print(f"Dataset: {data_config}")
    print(f"Split: {split}")
    print(f"Confidence threshold: {conf_threshold}")

    # Device selection
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    print("-" * 60)

    # Load model
    model = YOLO(str(model_path))

    # Run validation
    results = model.val(
        data=data_config,
        split=split,
        device=device,
        conf=conf_threshold,
        save_json=True,
        plots=True,
        verbose=True
    )

    # Extract metrics
    metrics = {
        "model": str(model_path),
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        },
        "per_class": {}
    }

    # Per-class metrics (for single class, just one entry)
    if hasattr(results.box, 'ap50') and len(results.box.ap50) > 0:
        metrics["per_class"]["license_plate"] = {
            "ap50": float(results.box.ap50[0]),
            "ap": float(results.box.ap[0]) if len(results.box.ap) > 0 else 0.0
        }

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nmAP@0.5:      {metrics['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['metrics']['mAP50-95']:.4f}")
    print(f"Precision:    {metrics['metrics']['precision']:.4f}")
    print(f"Recall:       {metrics['metrics']['recall']:.4f}")

    return metrics


def test_on_images(
    model_path: str,
    image_paths: List[str],
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.5,
    save: bool = True
) -> List[Dict]:
    """
    Test model on specific images.

    Args:
        model_path: Path to trained model
        image_paths: List of image paths to test
        output_dir: Output directory for predictions
        conf_threshold: Confidence threshold
        save: Whether to save prediction images

    Returns:
        List of prediction results
    """
    import torch
    import cv2
    from ultralytics import YOLO

    print("=" * 60)
    print("Testing on Images")
    print("=" * 60)

    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"

    results_list = []

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        print(f"\nProcessing: {img_path.name}")

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            device=device,
            save=save,
            project=output_dir,
            name="predictions",
            exist_ok=True
        )

        # Extract results
        for result in results:
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf[0].cpu().numpy())

                    detections.append({
                        "bbox": [int(x) for x in bbox],
                        "confidence": confidence
                    })

                    print(f"  Found plate: confidence={confidence:.3f}, "
                          f"bbox={[int(x) for x in bbox]}")

            results_list.append({
                "image": str(img_path),
                "detections": detections,
                "num_detections": len(detections)
            })

            if not detections:
                print("  No plates detected")

    return results_list


def benchmark_speed(
    model_path: str,
    num_iterations: int = 100,
    img_size: int = 640
) -> Dict:
    """
    Benchmark model inference speed.

    Args:
        model_path: Path to trained model
        num_iterations: Number of iterations for timing
        img_size: Image size for inference

    Returns:
        Speed benchmark results
    """
    import torch
    import numpy as np
    import time
    from ultralytics import YOLO

    print("=" * 60)
    print("Speed Benchmark")
    print("=" * 60)

    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    print(f"Image size: {img_size}x{img_size}")
    print(f"Iterations: {num_iterations}")

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        model.predict(dummy_img, verbose=False, device=device)

    # Benchmark
    print("Running benchmark...")
    times = []

    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        model.predict(dummy_img, verbose=False, device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    times = np.array(times)
    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "image_size": img_size,
        "iterations": num_iterations,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000 / np.mean(times))
    }

    print("\n" + "-" * 40)
    print(f"Mean inference time: {results['mean_ms']:.2f} ms")
    print(f"Std deviation:       {results['std_ms']:.2f} ms")
    print(f"Min time:            {results['min_ms']:.2f} ms")
    print(f"Max time:            {results['max_ms']:.2f} ms")
    print(f"FPS:                 {results['fps']:.1f}")

    return results


def generate_report(
    model_path: str,
    data_config: str,
    output_path: str,
    include_speed: bool = True
) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        model_path: Path to trained model
        data_config: Path to dataset config
        output_path: Output path for report
        include_speed: Include speed benchmark

    Returns:
        Path to generated report
    """
    report = {
        "model": str(model_path),
        "generated_at": datetime.now().isoformat(),
        "evaluation": {},
        "speed_benchmark": {}
    }

    # Evaluate on test set
    metrics = evaluate_model(model_path, data_config, split="test")
    report["evaluation"] = metrics

    # Speed benchmark
    if include_speed:
        speed = benchmark_speed(model_path)
        report["speed_benchmark"] = speed

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate license plate detection model"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate on test set")
    eval_parser.add_argument(
        "--model",
        type=str,
        default="ml/models/best.pt",
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        default="ml/training/configs/license_plate.yaml",
        help="Path to dataset config"
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on"
    )
    eval_parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test on specific images")
    test_parser.add_argument(
        "--model",
        type=str,
        default="ml/models/best.pt",
        help="Path to trained model"
    )
    test_parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Image paths to test"
    )
    test_parser.add_argument(
        "--output",
        type=str,
        default="ml/training/predictions",
        help="Output directory"
    )
    test_parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Speed benchmark")
    bench_parser.add_argument(
        "--model",
        type=str,
        default="ml/models/best.pt",
        help="Path to trained model"
    )
    bench_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations"
    )
    bench_parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate full report")
    report_parser.add_argument(
        "--model",
        type=str,
        default="ml/models/best.pt",
        help="Path to trained model"
    )
    report_parser.add_argument(
        "--data",
        type=str,
        default="ml/training/configs/license_plate.yaml",
        help="Path to dataset config"
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default="ml/training/evaluation_report.json",
        help="Output report path"
    )

    args = parser.parse_args()
    project_root = Path(__file__).parent.parent

    if args.command == "eval":
        model_path = project_root / args.model
        data_config = project_root / args.data
        evaluate_model(
            str(model_path),
            str(data_config),
            split=args.split,
            conf_threshold=args.conf
        )

    elif args.command == "test":
        model_path = project_root / args.model
        test_on_images(
            str(model_path),
            args.images,
            output_dir=str(project_root / args.output),
            conf_threshold=args.conf
        )

    elif args.command == "benchmark":
        model_path = project_root / args.model
        benchmark_speed(
            str(model_path),
            num_iterations=args.iterations,
            img_size=args.img_size
        )

    elif args.command == "report":
        model_path = project_root / args.model
        data_config = project_root / args.data
        output_path = project_root / args.output
        generate_report(
            str(model_path),
            str(data_config),
            str(output_path)
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
