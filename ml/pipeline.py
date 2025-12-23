"""
License Plate Recognition Pipeline

Combines detection and OCR for end-to-end license plate recognition.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO
import time
import uuid
import logging
from datetime import datetime

from .detector import LicensePlateDetector
from .ocr_engine import PlateOCR

logger = logging.getLogger(__name__)


class LPRPipeline:
    """
    Unified License Plate Recognition pipeline.

    Combines YOLO detection with EasyOCR for complete plate recognition.
    """

    def __init__(
        self,
        detector_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        ocr_languages: List[str] = None,
        use_gpu: bool = True,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the LPR pipeline.

        Args:
            detector_model_path: Path to YOLO weights file.
            confidence_threshold: Minimum detection confidence.
            ocr_languages: Languages for OCR (default: English + Arabic).
            use_gpu: Whether to use GPU acceleration.
            storage_path: Path to store processed images.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.storage_path = Path(storage_path) if storage_path else None

        logger.info("Initializing LPR Pipeline...")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        logger.info(f"Using GPU: {self.use_gpu}")

        # Initialize detector
        device = 'cuda' if self.use_gpu else 'cpu'
        self.detector = LicensePlateDetector(
            model_path=detector_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # Initialize OCR (English only)
        self.ocr = PlateOCR(
            languages=ocr_languages or ['en'],
            use_gpu=self.use_gpu
        )

        self.confidence_threshold = confidence_threshold

        logger.info("LPR Pipeline initialized successfully")

    def process_image(
        self,
        image_input: Union[np.ndarray, str, Path, BinaryIO],
        save_results: bool = False
    ) -> Dict:
        """
        Process an image and detect/read license plates.

        Args:
            image_input: Image as numpy array, file path, or file-like object.
            save_results: Whether to save cropped plates to storage.

        Returns:
            Dict with detection results:
                - success: bool
                - detection_id: unique ID
                - plates: list of detected plates
                - processing_time_ms: total processing time
                - image_size: original image dimensions
        """
        start_time = time.time()
        detection_id = str(uuid.uuid4())

        try:
            # Load image
            image = self._load_image(image_input)

            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'detection_id': detection_id
                }

            h, w = image.shape[:2]

            # Detect plates
            detections = self.detector.detect(image, return_crops=True)

            plates = []
            for i, detection in enumerate(detections):
                crop = detection.get('crop')

                # Run OCR on cropped plate
                if crop is not None and crop.size > 0:
                    plate_text, ocr_confidence = self.ocr.read_with_confidence(crop)
                else:
                    plate_text = ""
                    ocr_confidence = 0.0

                plate_info = {
                    'plate_number': plate_text,
                    'detection_confidence': detection['confidence'],
                    'ocr_confidence': ocr_confidence,
                    'combined_confidence': (detection['confidence'] + ocr_confidence) / 2,
                    'bbox': detection['bbox']
                }

                # Save cropped plate if requested
                if save_results and self.storage_path and crop is not None:
                    crop_filename = f"{detection_id}_plate_{i}.jpg"
                    crop_path = self._save_crop(crop, crop_filename)
                    plate_info['cropped_image_path'] = str(crop_path)

                plates.append(plate_info)

            processing_time = (time.time() - start_time) * 1000

            return {
                'success': True,
                'detection_id': detection_id,
                'plates': plates,
                'plate_count': len(plates),
                'processing_time_ms': round(processing_time, 2),
                'image_size': {'width': w, 'height': h},
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'detection_id': detection_id
            }

    def process_frame(
        self,
        frame: np.ndarray,
        draw_results: bool = True
    ) -> tuple:
        """
        Process a video frame for real-time detection.

        Args:
            frame: BGR video frame.
            draw_results: Whether to draw bounding boxes on frame.

        Returns:
            Tuple of (processed_frame, detections_list).
        """
        start_time = time.time()

        # Detect plates
        detections = self.detector.detect(frame, return_crops=True)

        results = []
        output_frame = frame.copy() if draw_results else frame

        for detection in detections:
            crop = detection.get('crop')
            bbox = detection['bbox']

            # Run OCR
            if crop is not None and crop.size > 0:
                plate_text, ocr_conf = self.ocr.read_with_confidence(crop)
            else:
                plate_text = ""
                ocr_conf = 0.0

            results.append({
                'plate_number': plate_text,
                'confidence': detection['confidence'],
                'ocr_confidence': ocr_conf,
                'bbox': bbox
            })

            # Draw on frame
            if draw_results:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)  # Green

                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{plate_text} ({detection['confidence']:.2f})"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 10, y1),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    output_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        return output_frame, results

    def process_batch(
        self,
        images: List[Union[np.ndarray, str, Path, BinaryIO]],
        save_results: bool = False
    ) -> List[Dict]:
        """
        Process multiple images.

        Args:
            images: List of images (arrays, paths, or file objects).
            save_results: Whether to save cropped plates.

        Returns:
            List of result dictionaries.
        """
        results = []
        for image in images:
            result = self.process_image(image, save_results=save_results)
            results.append(result)
        return results

    def _load_image(
        self,
        image_input: Union[np.ndarray, str, Path, BinaryIO]
    ) -> Optional[np.ndarray]:
        """
        Load image from various input types.

        Args:
            image_input: Image source.

        Returns:
            BGR numpy array or None if loading fails.
        """
        try:
            if isinstance(image_input, np.ndarray):
                return image_input

            elif isinstance(image_input, (str, Path)):
                path = str(image_input)
                image = cv2.imread(path)
                return image

            elif hasattr(image_input, 'read'):
                # File-like object (Flask FileStorage, BytesIO, etc.)
                file_bytes = image_input.read()
                if hasattr(image_input, 'seek'):
                    image_input.seek(0)  # Reset for potential re-read

                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image

            else:
                logger.error(f"Unsupported input type: {type(image_input)}")
                return None

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def _save_crop(self, crop: np.ndarray, filename: str) -> Path:
        """Save cropped plate image to storage."""
        if self.storage_path is None:
            raise ValueError("Storage path not configured")

        # Create date-based subdirectory
        today = datetime.now().strftime("%Y/%m/%d")
        save_dir = self.storage_path / today
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / filename
        cv2.imwrite(str(save_path), crop)

        return save_path

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update detection confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.detector.set_confidence_threshold(threshold)
        logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")

    def get_system_info(self) -> Dict:
        """Get system and model information."""
        info = {
            'gpu_available': torch.cuda.is_available(),
            'using_gpu': self.use_gpu,
            'confidence_threshold': self.confidence_threshold,
            'ocr_languages': self.ocr.get_supported_languages(),
        }

        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            info['gpu_memory_used'] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"

        return info
