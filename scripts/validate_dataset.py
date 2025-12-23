#!/usr/bin/env python3
"""
Validate License Plate Detection Dataset

Checks dataset integrity, displays sample annotations,
and provides statistics about the training data.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_yolo_annotation(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO format annotation.

    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))
    return annotations


def validate_annotation(annotation: Tuple, image_path: Path) -> List[str]:
    """
    Validate a single annotation.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    class_id, x_center, y_center, width, height = annotation

    # Check bounds
    if not (0 <= x_center <= 1):
        errors.append(f"x_center out of bounds: {x_center}")
    if not (0 <= y_center <= 1):
        errors.append(f"y_center out of bounds: {y_center}")
    if not (0 < width <= 1):
        errors.append(f"width out of bounds: {width}")
    if not (0 < height <= 1):
        errors.append(f"height out of bounds: {height}")

    # Check if box extends beyond image
    if x_center - width/2 < 0 or x_center + width/2 > 1:
        errors.append("Box extends beyond horizontal bounds")
    if y_center - height/2 < 0 or y_center + height/2 > 1:
        errors.append("Box extends beyond vertical bounds")

    # Check class ID
    if class_id < 0:
        errors.append(f"Invalid class ID: {class_id}")

    return errors


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


def validate_dataset(data_dir: Path, verbose: bool = False) -> Dict:
    """
    Validate entire dataset.

    Args:
        data_dir: Dataset root directory
        verbose: Print detailed errors

    Returns:
        Validation results dictionary
    """
    results = {
        "valid": True,
        "total_images": 0,
        "total_annotations": 0,
        "errors": [],
        "warnings": [],
        "splits": {}
    }

    for split in ["train", "val", "test"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        split_results = {
            "images": 0,
            "labels": 0,
            "annotations": 0,
            "matched": 0,
            "missing_labels": [],
            "orphan_labels": [],
            "annotation_errors": []
        }

        if not images_dir.exists():
            results["splits"][split] = split_results
            continue

        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = {f.stem: f for f in images_dir.iterdir()
                  if f.suffix.lower() in image_extensions}
        split_results["images"] = len(images)

        # Get all labels
        labels = {f.stem: f for f in labels_dir.iterdir()
                  if f.suffix.lower() == ".txt"} if labels_dir.exists() else {}
        split_results["labels"] = len(labels)

        # Check matching
        matched = set(images.keys()) & set(labels.keys())
        split_results["matched"] = len(matched)

        missing_labels = set(images.keys()) - set(labels.keys())
        orphan_labels = set(labels.keys()) - set(images.keys())

        split_results["missing_labels"] = list(missing_labels)[:10]  # Limit to 10
        split_results["orphan_labels"] = list(orphan_labels)[:10]

        if missing_labels:
            results["warnings"].append(
                f"{split}: {len(missing_labels)} images without labels"
            )

        if orphan_labels:
            results["warnings"].append(
                f"{split}: {len(orphan_labels)} orphan label files"
            )

        # Validate annotations
        for stem in matched:
            label_path = labels[stem]
            image_path = images[stem]

            try:
                annotations = load_yolo_annotation(label_path)
                split_results["annotations"] += len(annotations)

                for ann in annotations:
                    errors = validate_annotation(ann, image_path)
                    if errors:
                        split_results["annotation_errors"].append({
                            "file": label_path.name,
                            "errors": errors
                        })
                        if verbose:
                            for error in errors:
                                print(f"  {label_path.name}: {error}")

            except Exception as e:
                split_results["annotation_errors"].append({
                    "file": label_path.name,
                    "errors": [str(e)]
                })

        results["splits"][split] = split_results
        results["total_images"] += split_results["images"]
        results["total_annotations"] += split_results["annotations"]

    # Determine overall validity
    if results["warnings"] or any(
        s["annotation_errors"] for s in results["splits"].values()
    ):
        results["valid"] = False

    return results


def display_sample(data_dir: Path, split: str = "train", num_samples: int = 5):
    """
    Display sample images with annotations.

    Args:
        data_dir: Dataset root directory
        split: Data split to sample from
        num_samples: Number of samples to display
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not installed. Cannot display samples.")
        return

    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split

    if not images_dir.exists():
        print(f"Split '{split}' not found")
        return

    # Get image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in images_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {split}")
        return

    # Sample random images
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"\nDisplaying {len(samples)} samples from {split}:")
    print("-" * 40)

    for img_path in samples:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Load annotations
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = []
        if label_path.exists():
            annotations = load_yolo_annotation(label_path)

        print(f"\n{img_path.name}: {w}x{h}, {len(annotations)} annotations")

        # Draw annotations
        for ann in annotations:
            class_id, x_center, y_center, box_w, box_h = ann

            # Convert normalized coords to pixels
            x1 = int((x_center - box_w/2) * w)
            y1 = int((y_center - box_h/2) * h)
            x2 = int((x_center + box_w/2) * w)
            y2 = int((y_center + box_h/2) * h)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"plate", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"  - Box: ({x1}, {y1}) to ({x2}, {y2})")

        # Save annotated sample
        output_dir = data_dir.parent / "samples"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"sample_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        print(f"  Saved: {output_path}")


def print_statistics(results: Dict):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print(f"\nTotal images: {results['total_images']}")
    print(f"Total annotations: {results['total_annotations']}")

    if results['total_images'] > 0:
        avg_annotations = results['total_annotations'] / results['total_images']
        print(f"Average annotations per image: {avg_annotations:.2f}")

    print("\nPer-split breakdown:")
    print("-" * 40)

    for split, stats in results['splits'].items():
        if stats['images'] > 0:
            print(f"\n{split.upper()}:")
            print(f"  Images: {stats['images']}")
            print(f"  Labels: {stats['labels']}")
            print(f"  Matched: {stats['matched']}")
            print(f"  Annotations: {stats['annotations']}")

            if stats['missing_labels']:
                print(f"  Missing labels: {len(stats['missing_labels'])}")

            if stats['annotation_errors']:
                print(f"  Annotation errors: {len(stats['annotation_errors'])}")

    print("\n" + "=" * 60)

    if results['valid']:
        print("Validation: PASSED")
    else:
        print("Validation: WARNINGS/ERRORS FOUND")

        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")

        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate license plate detection dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory (default: ml/training/data)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages"
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Display sample images with annotations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to sample from (train/val/test)"
    )

    args = parser.parse_args()

    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "ml" / "training" / "data"

    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    print(f"Dataset directory: {data_dir}")

    if not data_dir.exists():
        print(f"\nError: Dataset directory not found: {data_dir}")
        print("Run 'python scripts/download_dataset.py' first")
        return 1

    # Validate dataset
    results = validate_dataset(data_dir, verbose=args.verbose)

    # Print statistics
    print_statistics(results)

    # Show samples if requested
    if args.show_samples:
        display_sample(data_dir, args.split, args.num_samples)

    return 0 if results['valid'] else 1


if __name__ == "__main__":
    sys.exit(main())
