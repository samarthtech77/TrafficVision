"""
Visualizer Module
Renders informative, high-quality overlays on traffic video frames:
- Dashboard HUD header banner
- Bounding boxes, vehicle labels, tracking IDs, speed magnitude
- Trajectory breadcrumb trails
- Lucas-Kanade Optical Flow motion vectors
- Reference counting line and crossing indicators
- Traffic condition badge
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class TrafficVisualizer:
    """
    Renders computer vision annotations and analytics HUD onto video frames.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("visualization", {}) if config else {}

        self.draw_boxes = cfg.get("draw_bounding_boxes", True)
        self.draw_ids = cfg.get("draw_tracking_ids", True)
        self.draw_trails = cfg.get("draw_trajectories", True)
        self.draw_line = cfg.get("draw_counting_line", True)
        self.draw_flow = cfg.get("draw_optical_flow", True)
        self.draw_hud = cfg.get("draw_hud_banner", True)

        colors = cfg.get("colors", {})
        self.color_bbox = tuple(colors.get("bbox", [0, 255, 127]))
        self.color_line = tuple(colors.get("line", [0, 0, 255]))
        self.color_line_active = tuple(colors.get("line_active", [0, 255, 255]))
        self.color_text = tuple(colors.get("text", [255, 255, 255]))
        self.color_hud_bg = tuple(colors.get("hud_bg", [30, 30, 30]))
        self.color_trail = tuple(colors.get("trail", [255, 191, 0]))
        self.color_flow = tuple(colors.get("flow_vector", [255, 0, 255]))

    def draw_annotation(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]],
        motion_stats: Dict[int, Dict[str, float]],
        traffic_stats: Dict[str, Any],
        counting_line: Tuple[Tuple[int, int], Tuple[int, int]],
        flow_vectors: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        newly_crossed_ids: Optional[set] = None,
    ) -> np.ndarray:
        """
        Draws all visual annotations onto the frame.

        :param frame: Source BGR image.
        :param tracks: Active vehicle tracks.
        :param motion_stats: Speed and motion vector statistics.
        :param traffic_stats: Aggregated traffic analytics (flow, density, condition).
        :param counting_line: (p1, p2) pixel coordinates of virtual counting line.
        :param flow_vectors: Optical flow vectors for drawing motion direction.
        :param newly_crossed_ids: Set of object_ids crossing line in current frame.
        :return: Annotated frame copy.
        """
        if frame is None:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Optical Flow Motion Vectors (Module 4)
        if self.draw_flow and flow_vectors:
            for (pt1, pt2) in flow_vectors:
                cv2.arrowedLine(canvas, pt1, pt2, self.color_flow, 1, tipLength=0.3)

        # 2. Draw Vehicle Trajectory Trails
        if self.draw_trails:
            for obj_id, track in tracks.items():
                history = track["history"]
                for i in range(1, len(history)):
                    pt1 = history[i - 1]
                    pt2 = history[i]
                    cv2.line(canvas, pt1, pt2, self.color_trail, 2, cv2.LINE_AA)

        # 3. Draw Bounding Boxes and Labels
        for obj_id, track in tracks.items():
            x, y, bw, bh = track["bbox"]
            label = track.get("label", "Vehicle")
            conf = track.get("confidence", 0.8)

            m_stat = motion_stats.get(obj_id, {})
            speed_kmh = m_stat.get("speed_kmh", 0.0)

            if self.draw_boxes:
                # Primary Bounding Box
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh), self.color_bbox, 2, cv2.LINE_AA)

                # Centroid marker
                cx, cy = track["centroid"]
                cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

            if self.draw_ids:
                # Text Label Header
                caption = f"ID:{obj_id} {label} {int(speed_kmh)}km/h"
                (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                # Background capsule for text legibility
                text_bg_y1 = max(0, y - th - 6)
                cv2.rectangle(canvas, (x, text_bg_y1), (x + tw + 6, y), (20, 20, 20), -1)
                cv2.putText(
                    canvas, caption, (x + 3, text_bg_y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.color_text, 1, cv2.LINE_AA
                )

        # 4. Draw Counting Line
        if self.draw_line and counting_line:
            p1, p2 = counting_line
            line_clr = self.color_line_active if (newly_crossed_ids and len(newly_crossed_ids) > 0) else self.color_line
            thickness = 4 if (newly_crossed_ids and len(newly_crossed_ids) > 0) else 2
            cv2.line(canvas, p1, p2, line_clr, thickness, cv2.LINE_AA)

            # Counting line label
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                canvas, "--- VIRTUAL COUNTING GATE ---", (mid_x - 110, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_clr, 1, cv2.LINE_AA
            )

        # 5. Draw HUD Dashboard Banner
        if self.draw_hud:
            self._draw_hud_banner(canvas, traffic_stats, w, h)

        return canvas

    def _draw_hud_banner(self, canvas: np.ndarray, stats: Dict[str, Any], w: int, h: int) -> None:
        """Renders top overlay dashboard with live traffic parameters."""
        hud_height = 55
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), self.color_hud_bg, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, hud_height), (w, hud_height), (100, 100, 100), 1)

        # Title
        cv2.putText(
            canvas, "TrafficVision | CSE3010 BYOP", (15, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Metrics text
        total = stats.get("total_count", 0)
        flow = stats.get("flow_rate_vpm", 0.0)
        active = stats.get("active_vehicles", 0)
        density_cat = stats.get("density_category", "LOW")
        condition = stats.get("traffic_condition", "LOW")
        cond_color = stats.get("condition_color", (0, 255, 0))

        metrics_str = f"Count: {total} | Flow: {flow:.1f} v/min | Active Density: {active} ({density_cat})"
        cv2.putText(
            canvas, metrics_str, (15, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Traffic Condition Badge
        badge_text = f"CONDITION: {condition}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bx = w - tw - 25
        by = 12

        cv2.rectangle(canvas, (bx - 8, by), (w - 10, by + th + 14), cond_color, -1)
        cv2.putText(
            canvas, badge_text, (bx, by + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
        )
