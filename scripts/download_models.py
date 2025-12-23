#!/usr/bin/env python3
"""
Download pre-trained models for License Plate Recognition.

This script downloads:
1. YOLOv8 base model
2. EasyOCR language models (English, Arabic)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_yolo_model():
    """Download YOLOv8 model."""
    print("Downloading YOLOv8m model (medium - better accuracy)...")

    from ultralytics import YOLO

    # Download and cache YOLOv8m (medium model for better accuracy)
    model = YOLO('yolov8m.pt')

    # Save to models directory
    models_dir = Path(__file__).parent.parent / 'ml' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if trained model exists
    trained_model = models_dir / 'best.pt'
    if trained_model.exists():
        print(f"Trained license plate model found: {trained_model}")
        print("This trained model will be used for detection.")
    else:
        print(f"YOLOv8m base model downloaded and cached")
        print(f"Models directory: {models_dir}")
        print("\nNote: For better license plate detection accuracy, train a custom model:")
        print("  1. python scripts/download_dataset.py")
        print("  2. python scripts/train_detector.py")

    return True


def download_ocr_models():
    """Download EasyOCR models for English."""
    print("\nDownloading EasyOCR models (English only)...")

    import easyocr

    # This will download the models on first use
    reader = easyocr.Reader(['en'], gpu=False, verbose=True)

    print("EasyOCR English models downloaded successfully")
    return True


def main():
    print("=" * 60)
    print("License Plate Recognition - Model Downloader")
    print("=" * 60)

    success = True

    # Download YOLO
    try:
        download_yolo_model()
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        success = False

    # Download OCR models
    try:
        download_ocr_models()
    except Exception as e:
        print(f"Error downloading OCR models: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("All models downloaded successfully!")
    else:
        print("Some models failed to download. Check errors above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
