"""
Vehicle Detector Module
Provides object detection implementations:
1. Classical Background Subtraction Detector (MOG2 + Morphological Filtering + Contours)
   - Strongly aligned with CSE3010 Syllabus Module 3 & Module 4
2. Haar Cascade Classifier Detector (Optional Classical ML)

Returns structured detections: list of tuples (x, y, w, h, confidence, class_name)
"""

import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("TrafficVision.Detector")

# Type alias for detection: (x, y, width, height, confidence_score, class_label)
Detection = Tuple[int, int, int, int, float, str]


class BaseDetector(ABC):
    """Abstract Base Class for vehicle detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Detect vehicles in a frame.

        :param frame: Raw BGR video frame.
        :param preprocessed: Optional preprocessed frame (grayscale/blurred/enhanced).
        :return: List of detections [(x, y, w, h, confidence, class_name), ...]
        """
        pass


class BackgroundSubtractorDetector(BaseDetector):
    """
    Classical Computer Vision Detector using MOG2 Background Subtraction,
    Morphological Filtering, and Contour Analysis.
    Aligned with CSE3010 Syllabus Module 3 (Contours/Morphology) & Module 4 (Background Subtraction).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("detector", {}).get("bg_subtractor", {}) if config else {}

        history = cfg.get("history", 500)
        var_threshold = cfg.get("var_threshold", 40)
        detect_shadows = cfg.get("detect_shadows", True)

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

        self.shadow_value = cfg.get("shadow_value", 127)
        self.morph_kernel_size = cfg.get("morph_kernel_size", 5)
        self.min_contour_area = cfg.get("min_contour_area", 500)
        self.max_contour_area = cfg.get("max_contour_area", 35000)
        self.min_aspect_ratio = cfg.get("min_aspect_ratio", 0.3)
        self.max_aspect_ratio = cfg.get("max_aspect_ratio", 3.5)

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )

    def detect(self, frame: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> List[Detection]:
        if frame is None:
            return []

        input_img = preprocessed if preprocessed is not None else frame

        # 1. Background Subtraction
        fg_mask = self.subtractor.apply(input_img)

        # 2. Remove shadows (shadow pixels are marked as ~127 by MOG2)
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 3. Morphological Operations (Module 3 Filtering concept)
        # Opening: removes small noise speckles
        opening = cv2.morphologyEx(fg_thresh, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # Closing: connects broken vehicle shapes and fills interior holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # Dilation: expands contour boundaries slightly for accurate bounding box
        dilated = cv2.dilate(closing, self.kernel, iterations=1)

        # 4. Contour Extraction & Filtering (Module 3 Contour Segmentation)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0.0

                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    # Estimate confidence based on blob density ratio within bounding box
                    bbox_area = w * h
                    confidence = min(0.99, max(0.50, float(area) / bbox_area if bbox_area > 0 else 0.70))

                    # Classify roughly by bounding box size (Car vs Bus/Truck)
                    class_label = "Vehicle"
                    if area > 12000:
                        class_label = "Bus/Truck"
                    elif aspect_ratio < 0.6:
                        class_label = "Motorcycle"

                    detections.append((x, y, w, h, round(confidence, 2), class_label))

        return detections


class CascadeDetector(BaseDetector):
    """
    OpenCV Cascade Classifier (Haar / LBP) Detector for Vehicle Detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("detector", {}).get("cascade", {}) if config else {}
        xml_path = cfg.get("xml_path", "config/haarcascade_car.xml")
        self.scale_factor = cfg.get("scale_factor", 1.1)
        self.min_neighbors = cfg.get("min_neighbors", 3)
        self.min_size = tuple(cfg.get("min_size", [30, 30]))

        if not os.path.exists(xml_path):
            logger.warning(f"Cascade XML file '{xml_path}' not found. Falling back to Background Subtractor.")
            self.cascade = None
        else:
            self.cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, frame: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> List[Detection]:
        if self.cascade is None or self.cascade.empty():
            return []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        detections: List[Detection] = []
        for (x, y, w, h) in rects:
            detections.append((int(x), int(y), int(w), int(h), 0.85, "Vehicle"))

        return detections


def get_detector(detector_type: str = "bg_subtractor", config: Optional[Dict[str, Any]] = None) -> BaseDetector:
    """
    Factory function to instantiate object detector.

    :param detector_type: 'bg_subtractor' or 'cascade'
    :param config: Configuration dictionary.
    :return: An instance of BaseDetector.
    """
    dt = detector_type.lower()
    if dt in ("bg_subtractor", "mog2", "default"):
        return BackgroundSubtractorDetector(config)
    elif dt in ("cascade", "haar"):
        detector = CascadeDetector(config)
        if detector.cascade is None or detector.cascade.empty():
            logger.warning("Cascade detector unavailable, defaulting to BackgroundSubtractorDetector.")
            return BackgroundSubtractorDetector(config)
        return detector
    else:
        logger.warning(f"Unknown detector type '{detector_type}', defaulting to BackgroundSubtractorDetector.")
        return BackgroundSubtractorDetector(config)
