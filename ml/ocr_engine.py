"""
License Plate OCR Engine using EasyOCR

Supports multi-language text recognition including Arabic and Latin characters.
"""

import cv2
import numpy as np
import easyocr
import re
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PlateOCR:
    """
    OCR engine for license plate text recognition.

    Supports Arabic and Latin characters with GPU acceleration.
    """

    def __init__(
        self,
        languages: List[str] = None,
        use_gpu: bool = True,
        model_storage_directory: Optional[str] = None
    ):
        """
        Initialize the OCR engine.

        Args:
            languages: List of language codes. Default: ['en', 'ar'] for English and Arabic.
            use_gpu: Whether to use GPU acceleration.
            model_storage_directory: Custom directory for storing OCR models.
        """
        # Default to English and Arabic for universal plate support
        self.languages = languages or ['en', 'ar']

        logger.info(f"Initializing EasyOCR with languages: {self.languages}")
        logger.info(f"GPU enabled: {use_gpu}")

        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            model_storage_directory=model_storage_directory,
            verbose=False
        )

        # Common OCR corrections for license plates
        self.char_corrections = {
            'O': '0',  # O to zero
            'I': '1',  # I to one
            'Z': '2',  # Z to two
            'S': '5',  # S to five
            'B': '8',  # B to eight
            'G': '6',  # G to six
        }

        logger.info("OCR engine initialized successfully")

    def read(
        self,
        plate_image: np.ndarray,
        preprocess: bool = True,
        detail: int = 0
    ) -> str:
        """
        Read text from a license plate image.

        Args:
            plate_image: BGR or grayscale image of the plate.
            preprocess: Whether to apply preprocessing.
            detail: 0 = text only, 1 = include bounding boxes and confidence.

        Returns:
            Recognized plate text (cleaned and formatted).
        """
        if plate_image is None or plate_image.size == 0:
            return ""

        # Preprocess image
        if preprocess:
            processed = self._preprocess(plate_image)
        else:
            processed = plate_image

        # Run OCR
        try:
            results = self.reader.readtext(processed, detail=detail)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""

        if not results:
            return ""

        # Extract text
        if detail == 0:
            text = ' '.join(results)
        else:
            # detail=1 returns list of (bbox, text, confidence)
            text = ' '.join([r[1] for r in results])

        # Clean and format
        cleaned = self._clean_text(text)

        return cleaned

    def read_with_confidence(
        self,
        plate_image: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[str, float]:
        """
        Read text with confidence score.

        Args:
            plate_image: BGR or grayscale image of the plate.
            preprocess: Whether to apply preprocessing.

        Returns:
            Tuple of (plate_text, average_confidence).
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        if preprocess:
            processed = self._preprocess(plate_image)
        else:
            processed = plate_image

        try:
            results = self.reader.readtext(processed)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

        if not results:
            return "", 0.0

        # Extract text and confidence
        texts = []
        confidences = []

        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        cleaned = self._clean_text(combined_text)

        return cleaned, avg_confidence

    def read_detailed(
        self,
        plate_image: np.ndarray,
        preprocess: bool = True
    ) -> List[Dict]:
        """
        Read text with detailed information including bounding boxes.

        Args:
            plate_image: BGR or grayscale image of the plate.
            preprocess: Whether to apply preprocessing.

        Returns:
            List of dicts with 'text', 'confidence', 'bbox' keys.
        """
        if plate_image is None or plate_image.size == 0:
            return []

        if preprocess:
            processed = self._preprocess(plate_image)
        else:
            processed = plate_image

        try:
            results = self.reader.readtext(processed)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []

        detailed_results = []
        for bbox, text, conf in results:
            detailed_results.append({
                'text': self._clean_text(text),
                'raw_text': text,
                'confidence': float(conf),
                'bbox': [[int(p[0]), int(p[1])] for p in bbox]
            })

        return detailed_results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Preprocessed grayscale image.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize for better accuracy (width should be at least 200px)
        height, width = gray.shape
        if width < 200:
            scale = 200 / width
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        elif width > 800:
            scale = 800 / width
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        # Denoise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Optional: adaptive threshold for difficult cases
        # thresh = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2
        # )

        return gray

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize plate text.

        Args:
            text: Raw OCR text.

        Returns:
            Cleaned plate number.
        """
        if not text:
            return ""

        # Remove unwanted characters (keep alphanumeric, Arabic, spaces, dashes)
        # Arabic Unicode range: \u0600-\u06FF
        cleaned = re.sub(r'[^\w\u0600-\u06FF\s\-]', '', text)

        # Uppercase Latin characters
        cleaned = cleaned.upper()

        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())

        # Apply common corrections for numeric contexts
        # Only apply to purely Latin strings
        if re.match(r'^[A-Z0-9\s\-]+$', cleaned):
            result = []
            for char in cleaned:
                if char in self.char_corrections:
                    # Check context - convert to number if surrounded by numbers
                    result.append(char)  # Keep original for now
                else:
                    result.append(char)
            cleaned = ''.join(result)

        return cleaned.strip()

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.languages
