"""
License Plate Recognition ML Module

This module provides:
- License plate detection using YOLOv8
- Text recognition using EasyOCR (supports Arabic + Latin)
- Unified pipeline for end-to-end processing
"""

from .pipeline import LPRPipeline
from .detector import LicensePlateDetector
from .ocr_engine import PlateOCR

__all__ = ['LPRPipeline', 'LicensePlateDetector', 'PlateOCR']
