"""
Unit tests for ImagePreprocessor module.
"""

import pytest
import numpy as np
from src.preprocessor import ImagePreprocessor


def test_preprocessor_grayscale_blur():
    config = {
        "preprocessing": {
            "grayscale": True,
            "gaussian_blur": True,
            "blur_kernel": [5, 5],
            "clahe_enable": True,
        }
    }
    preprocessor = ImagePreprocessor(config)
    dummy_frame = np.full((200, 300, 3), 120, dtype=np.uint8)

    processed = preprocessor.process(dummy_frame)
    assert len(processed.shape) == 2  # Grayscale image
    assert processed.shape == (200, 300)


def test_preprocessor_roi_mask():
    config = {
        "preprocessing": {
            "grayscale": True,
            "roi_mask_enable": True,
            "roi_vertices": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        }
    }
    preprocessor = ImagePreprocessor(config)
    dummy_frame = np.full((100, 100, 3), 255, dtype=np.uint8)

    processed = preprocessor.process(dummy_frame)
    # Outside ROI should be masked to 0
    assert processed[5, 5] == 0
    # Inside ROI (e.g. at 50, 50) should retain non-zero values
    assert processed[50, 50] > 0
