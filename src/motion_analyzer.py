"""
Motion Analyzer Module
Computes vehicle velocity vectors and sparse Lucas-Kanade (KLT) Optical Flow
for spatio-temporal motion parameter estimation.
Aligned with CSE3010 Syllabus Module 4 (Optical Flow, KLT, Motion Parameter Estimation).
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MotionAnalyzer:
    """
    Computes optical flow vectors and per-vehicle motion parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("motion", {}) if config else {}
        self.enable_klt = cfg.get("enable_klt_flow", True)
        self.max_corners = cfg.get("max_corners", 100)
        self.quality_level = cfg.get("quality_level", 0.3)
        self.min_distance = cfg.get("min_distance", 7)
        self.block_size = cfg.get("block_size", 7)

        # Lucas-Kanade Optical Flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.flow_vectors: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]],
        fps: float = 25.0,
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculates motion vector parameters for each tracked vehicle.

        :param frame: Current BGR frame.
        :param tracks: Active tracks from CentroidTracker.
        :param fps: Video FPS.
        :return: Dictionary mapping object_id to motion stats {'speed_px': float, 'angle_deg': float, 'speed_kmh': float}.
        """
        motion_stats: Dict[int, Dict[str, float]] = {}

        if frame is None:
            return motion_stats

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.flow_vectors = []

        # 1. Compute per-object centroid displacement velocity from track history
        for obj_id, track in tracks.items():
            history = track["history"]
            if len(history) >= 2:
                p_prev = history[-2]
                p_curr = history[-1]

                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]

                speed_px = float(np.sqrt(dx**2 + dy**2))
                angle_rad = np.arctan2(dy, dx)
                angle_deg = float(np.degrees(angle_rad))

                # Heuristic pixel-to-km/h scale factor for standard video views
                speed_kmh = min(120.0, speed_px * (fps / 25.0) * 3.6 * 0.5)

                motion_stats[obj_id] = {
                    "speed_px": round(speed_px, 2),
                    "angle_deg": round(angle_deg, 1),
                    "speed_kmh": round(speed_kmh, 1),
                    "dx": dx,
                    "dy": dy,
                }
            else:
                motion_stats[obj_id] = {
                    "speed_px": 0.0,
                    "angle_deg": 0.0,
                    "speed_kmh": 0.0,
                    "dx": 0,
                    "dy": 0,
                }

        # 2. Sparse Lucas-Kanade Optical Flow inside active vehicle bounding boxes
        if self.enable_klt and self.prev_gray is not None and len(tracks) > 0:
            for obj_id, track in tracks.items():
                x, y, w, h = track["bbox"]

                # Ensure ROI stays within frame boundaries
                H, W = gray.shape[:2]
                rx1, ry1 = max(0, x), max(0, y)
                rx2, ry2 = min(W, x + w), min(H, y + h)

                if (rx2 - rx1) > 10 and (ry2 - ry1) > 10:
                    prev_roi = self.prev_gray[ry1:ry2, rx1:rx2]

                    # Extract Shi-Tomasi corners in ROI
                    p0 = cv2.goodFeaturesToTrack(
                        prev_roi,
                        maxCorners=10,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance,
                        blockSize=self.block_size,
                    )

                    if p0 is not None and len(p0) > 0:
                        # Shift corner coordinates back to frame coordinates
                        p0[:, 0, 0] += rx1
                        p0[:, 0, 1] += ry1

                        # Calculate Lucas-Kanade Optical Flow
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, gray, p0, None, **self.lk_params
                        )

                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            for new, old in zip(good_new, good_old):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                self.flow_vectors.append(((int(c), int(d)), (int(a), int(b))))

        self.prev_gray = gray.copy()
        return motion_stats

    def get_flow_vectors(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Returns computed optical flow vector line segments for drawing."""
        return self.flow_vectors
