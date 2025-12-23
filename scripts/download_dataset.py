#!/usr/bin/env python3
"""
Download License Plate Detection Dataset

Downloads European license plate datasets from Roboflow Universe
and organizes them for YOLOv8 training.
"""

import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Roboflow public datasets for license plates
DATASETS = {
    "license-plate-recognition": {
        "url": "https://universe.roboflow.com/ds/Ej9hVJVLcT?key=open",
        "workspace": "roboflow-universe-projects",
        "project": "license-plate-recognition-rxg4e",
        "version": 4,
        "description": "10,125 license plate images in YOLO format"
    },
    "vehicle-registration-plates": {
        "url": "https://universe.roboflow.com/ds/VehiclePlates",
        "workspace": "augmented-startups",
        "project": "vehicle-registration-plates-trudk",
        "version": 2,
        "description": "European vehicle registration plates"
    }
}


def download_from_roboflow(
    api_key: Optional[str],
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    format: str = "yolov8"
) -> bool:
    """
    Download dataset from Roboflow.

    Args:
        api_key: Roboflow API key (optional for public datasets)
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        output_dir: Output directory
        format: Export format (default: yolov8)

    Returns:
        True if successful
    """
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key or "")
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(format, location=str(output_dir))

        print(f"Downloaded to: {output_dir}")
        return True

    except ImportError:
        print("Roboflow package not installed. Installing...")
        os.system(f"{sys.executable} -m pip install roboflow")
        return download_from_roboflow(api_key, workspace, project, version, output_dir, format)

    except Exception as e:
        print(f"Error downloading from Roboflow: {e}")
        return False


def download_open_images_plates(output_dir: Path) -> bool:
    """
    Download license plate subset from Open Images Dataset.
    Uses FiftyOne for efficient downloading.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        # Download Open Images with license plate class
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=["Vehicle registration plate"],
            max_samples=5000
        )

        # Export in YOLO format
        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections"
        )

        print(f"Downloaded Open Images plates to: {output_dir}")
        return True

    except ImportError:
        print("FiftyOne not installed. Skipping Open Images download.")
        print("Install with: pip install fiftyone")
        return False
    except Exception as e:
        print(f"Error downloading Open Images: {e}")
        return False


def organize_dataset(source_dir: Path, target_dir: Path) -> dict:
    """
    Organize downloaded dataset into standard YOLO structure.

    Args:
        source_dir: Downloaded dataset directory
        target_dir: Target training data directory

    Returns:
        Statistics dict with counts
    """
    stats = {"train": 0, "val": 0, "test": 0}

    # Standard Roboflow structure
    for split in ["train", "valid", "test"]:
        # Map 'valid' to 'val' for consistency
        target_split = "val" if split == "valid" else split

        src_images = source_dir / split / "images"
        src_labels = source_dir / split / "labels"

        if not src_images.exists():
            continue

        tgt_images = target_dir / "images" / target_split
        tgt_labels = target_dir / "labels" / target_split

        tgt_images.mkdir(parents=True, exist_ok=True)
        tgt_labels.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_file in src_images.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                shutil.copy2(img_file, tgt_images / img_file.name)
                stats[target_split] += 1

                # Copy corresponding label
                label_name = img_file.stem + ".txt"
                label_file = src_labels / label_name
                if label_file.exists():
                    shutil.copy2(label_file, tgt_labels / label_name)

    return stats


def verify_dataset(data_dir: Path) -> dict:
    """
    Verify dataset integrity.

    Args:
        data_dir: Dataset directory

    Returns:
        Verification results
    """
    results = {
        "valid": True,
        "errors": [],
        "stats": {}
    }

    for split in ["train", "val", "test"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists():
            results["stats"][split] = {"images": 0, "labels": 0, "matched": 0}
            continue

        images = set(f.stem for f in images_dir.glob("*")
                     if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"])
        labels = set(f.stem for f in labels_dir.glob("*.txt"))

        matched = images & labels
        missing_labels = images - labels
        orphan_labels = labels - images

        results["stats"][split] = {
            "images": len(images),
            "labels": len(labels),
            "matched": len(matched)
        }

        if missing_labels:
            results["errors"].append(f"{split}: {len(missing_labels)} images without labels")

        if orphan_labels:
            results["errors"].append(f"{split}: {len(orphan_labels)} orphan label files")

    if results["errors"]:
        results["valid"] = False

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download license plate detection datasets"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ml/training/data)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="license-plate-recognition",
        help="Dataset to download"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip dataset verification"
    )

    args = parser.parse_args()

    # Set paths
    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "ml" / "training" / "data"
    temp_dir = project_root / "ml" / "training" / ".download_temp"

    print("=" * 60)
    print("License Plate Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Download datasets
    datasets_to_download = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    total_stats = {"train": 0, "val": 0, "test": 0}

    for dataset_name in datasets_to_download:
        dataset_info = DATASETS[dataset_name]
        print(f"\nDownloading: {dataset_name}")
        print(f"Description: {dataset_info['description']}")

        # Create temp directory for this dataset
        dataset_temp = temp_dir / dataset_name
        dataset_temp.mkdir(parents=True, exist_ok=True)

        success = download_from_roboflow(
            api_key=args.api_key,
            workspace=dataset_info["workspace"],
            project=dataset_info["project"],
            version=dataset_info["version"],
            output_dir=dataset_temp
        )

        if success:
            # Find the actual download directory (Roboflow creates a subdirectory)
            subdirs = list(dataset_temp.glob("*"))
            actual_dir = subdirs[0] if subdirs else dataset_temp

            # Organize into target structure
            stats = organize_dataset(actual_dir, output_dir)

            for split, count in stats.items():
                total_stats[split] += count

            print(f"Organized: train={stats['train']}, val={stats['val']}, test={stats['test']}")
        else:
            print(f"Failed to download {dataset_name}")

    # Cleanup temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total images: {sum(total_stats.values())}")
    print(f"  - Train: {total_stats['train']}")
    print(f"  - Validation: {total_stats['val']}")
    print(f"  - Test: {total_stats['test']}")

    # Verify dataset
    if not args.skip_verify:
        print("\nVerifying dataset...")
        results = verify_dataset(output_dir)

        if results["valid"]:
            print("Dataset verification: PASSED")
        else:
            print("Dataset verification: FAILED")
            for error in results["errors"]:
                print(f"  - {error}")

    print("\n" + "=" * 60)
    print("Dataset ready for training!")
    print(f"Config file: ml/training/configs/license_plate.yaml")
    print("Run training with: python scripts/train_detector.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
