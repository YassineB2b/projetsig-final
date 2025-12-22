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
    print("Downloading YOLOv8n model...")

    from ultralytics import YOLO

    # Download and cache YOLOv8n
    model = YOLO('yolov8n.pt')

    # Save to models directory
    models_dir = Path(__file__).parent.parent / 'ml' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # The model is automatically cached by ultralytics
    print(f"YOLOv8n model downloaded and cached")
    print(f"Models directory: {models_dir}")

    return True


def download_ocr_models():
    """Download EasyOCR models for English and Arabic."""
    print("\nDownloading EasyOCR models...")

    import easyocr

    # This will download the models on first use
    reader = easyocr.Reader(['en', 'ar'], gpu=False, verbose=True)

    print("EasyOCR models downloaded successfully")
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
