"""Detector adapters for AGCM-Net."""
from .base import Detector
from .yolo import YOLODetector
from .picodet import PicoDetDetector
from .rtmdet import RTMDetDetector
from .ort_picodet import ORTPicoDetDetector
from .deimv2 import DEIMv2Adapter

__all__ = ['Detector', 'YOLODetector', 'PicoDetDetector',
           'RTMDetDetector', 'ORTPicoDetDetector', 'DEIMv2Adapter']
