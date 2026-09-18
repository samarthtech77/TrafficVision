"""
Image Preprocessor Module
Applies Computer Vision preprocessing operations aligned with CSE3010:
- Grayscale transformation
- Gaussian blurring (smoothing)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Region of Interest (ROI) polygonal masking
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class ImagePreprocessor:
    """
    Encapsulates image preprocessing routines for traffic video frames.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration settings.
        """
        self.config = config or {}
        prep_cfg = self.config.get("preprocessing", {})

        self.grayscale = prep_cfg.get("grayscale", True)
        self.gaussian_blur = prep_cfg.get("gaussian_blur", True)
        self.blur_kernel = tuple(prep_cfg.get("blur_kernel", [5, 5]))

        self.clahe_enable = prep_cfg.get("clahe_enable", True)
        clip_limit = prep_cfg.get("clahe_clip_limit", 2.0)
        grid_size = tuple(prep_cfg.get("clahe_grid_size", [8, 8]))
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        self.roi_mask_enable = prep_cfg.get("roi_mask_enable", False)
        self.roi_vertices = prep_cfg.get("roi_vertices", [])

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a raw BGR frame and returns a preprocessed image (grayscale or enhanced BGR).

        :param frame: Raw input frame (BGR uint8 array).
        :return: Preprocessed frame.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        h, w = frame.shape[:2]
        processed = frame.copy()

        # 1. Grayscale Conversion (Module 1 / Preprocessing concept)
        if self.grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # 2. CLAHE (Histogram Enhancement concept)
            if self.clahe_enable:
                processed = self.clahe.apply(processed)

            # 3. Gaussian Blurring (Filtering & Smoothing concept)
            if self.gaussian_blur:
                processed = cv2.GaussianBlur(processed, self.blur_kernel, 0)
        else:
            if self.gaussian_blur:
                processed = cv2.GaussianBlur(processed, self.blur_kernel, 0)

        # 4. ROI Polygonal Masking
        if self.roi_mask_enable and len(self.roi_vertices) >= 3:
            mask = np.zeros((h, w), dtype=np.uint8) if self.grayscale else np.zeros((h, w, 3), dtype=np.uint8)
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in self.roi_vertices],
                dtype=np.int32,
            )
            pts = pts.reshape((-1, 1, 2))

            if self.grayscale:
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.fillPoly(mask, [pts], (255, 255, 255))

            processed = cv2.bitwise_and(processed, mask)

        return processed

    @staticmethod
    def get_roi_mask(shape: Tuple[int, int], vertices: list) -> np.ndarray:
        """
        Creates a binary mask (255 inside ROI, 0 outside) from normalized vertices.
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(vertices) >= 3:
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in vertices],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask.fill(255)
        return mask
