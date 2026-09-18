"""
Unit tests for Vehicle Detector module.
"""

import pytest
import numpy as np
import cv2
from src.detector import BackgroundSubtractorDetector, get_detector


def test_bg_subtractor_detector():
    detector = BackgroundSubtractorDetector(config={
        "detector": {
            "bg_subtractor": {
                "min_contour_area": 100,
                "max_contour_area": 5000,
            }
        }
    })

    # Train background with blank black frame
    blank_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    for _ in range(5):
        detector.detect(blank_frame)

    # Frame with a moving bright vehicle box
    vehicle_frame = blank_frame.copy()
    cv2.rectangle(vehicle_frame, (100, 100), (160, 150), (255, 255, 255), -1)

    detections = detector.detect(vehicle_frame)
    assert isinstance(detections, list)
    if len(detections) > 0:
        x, y, w, h, conf, label = detections[0]
        assert w > 0 and h > 0
        assert 0.0 <= conf <= 1.0


def test_detector_factory():
    d1 = get_detector("bg_subtractor")
    assert isinstance(d1, BackgroundSubtractorDetector)
