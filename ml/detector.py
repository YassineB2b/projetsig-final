"""
License Plate Detector using YOLOv8

Provides GPU-accelerated detection of license plates in images.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """YOLO-based license plate detector with GPU support."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the license plate detector.

        Args:
            model_path: Path to YOLO weights file. If None, uses default YOLOv8n.
            confidence_threshold: Minimum confidence for detections (0-1).
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing detector on device: {self.device}")

        # Load model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading custom model from: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Use pre-trained YOLOv8n as base (will download if needed)
            logger.info("Loading YOLOv8n base model")
            self.model = YOLO('yolov8n.pt')

        self.model.to(self.device)
        self._warmup()

    def _warmup(self) -> None:
        """Warmup model for faster first inference."""
        logger.info("Warming up detector...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        logger.info("Detector warmup complete")

    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = False
    ) -> List[Dict]:
        """
        Detect license plates in an image.

        Args:
            image: BGR image as numpy array.
            return_crops: If True, include cropped plate images in results.

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: detection confidence (0-1)
                - crop: cropped plate image (if return_crops=True)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                confidence = float(box.conf[0].cpu().numpy())

                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': int(box.cls[0].cpu().numpy())
                }

                # Add cropped image if requested
                if return_crops:
                    x1, y1, x2, y2 = bbox
                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    detection['crop'] = image[y1:y2, x1:x2].copy()

                detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        return_crops: bool = False
    ) -> List[List[Dict]]:
        """
        Detect license plates in multiple images.

        Args:
            images: List of BGR images.
            return_crops: If True, include cropped plate images.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        # Process in batches for efficiency
        results = self.model(
            images,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )

        for i, result in enumerate(results):
            image_detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    confidence = float(box.conf[0].cpu().numpy())

                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': int(box.cls[0].cpu().numpy())
                    }

                    if return_crops:
                        x1, y1, x2, y2 = bbox
                        image = images[i]
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        detection['crop'] = image[y1:y2, x1:x2].copy()

                    image_detections.append(detection)

            image_detections.sort(key=lambda x: x['confidence'], reverse=True)
            all_detections.append(image_detections)

        return all_detections

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device == 'cuda' and torch.cuda.is_available()

    def get_device_info(self) -> Dict:
        """Get information about the current device."""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            info['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"

        return info
