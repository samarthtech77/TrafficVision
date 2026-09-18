"""
Vehicle Counter Module
Implements robust line-crossing detection using vector cross-products
and trajectory segment intersection testing.
Prevents duplicate vehicle counting and tracks directional counts.
"""

from typing import Dict, Tuple, List, Set, Any, Optional


class VehicleCounter:
    """
    Detects when tracked vehicle trajectories cross a virtual reference line.
    """

    def __init__(
        self,
        line_coords: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.05, 0.55), (0.95, 0.55)),
        direction: str = "both",
    ):
        """
        Initialize VehicleCounter.

        :param line_coords: ((x1, y1), (x2, y2)) normalized or absolute line endpoints.
        :param direction: Allowed counting direction ('both', 'down', 'up').
        """
        self.norm_line = line_coords
        self.direction = direction.lower()

        self.counted_ids: Set[int] = set()
        self.total_count: int = 0
        self.counts_by_direction: Dict[str, int] = {"down": 0, "up": 0}
        self.counts_by_class: Dict[str, int] = {}
        self.recent_crossings: List[Dict[str, Any]] = []

    def get_absolute_line(self, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert normalized line coordinates to absolute pixel coordinates."""
        (x1, y1), (x2, y2) = self.norm_line
        p1 = (int(x1 * width) if x1 <= 1.0 else int(x1), int(y1 * height) if y1 <= 1.0 else int(y1))
        p2 = (int(x2 * width) if x2 <= 1.0 else int(x2), int(y2 * height) if y2 <= 1.0 else int(y2))
        return p1, p2

    @staticmethod
    def _ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> float:
        """
        Calculates 2D cross-product orientation for 3 points A, B, C.
        Returns > 0 if counter-clockwise, < 0 if clockwise, 0 if collinear.
        """
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
        """
        Checks if line segment AB (counting line) intersects trajectory segment CD.
        """
        ccw1 = self._ccw(A, B, C)
        ccw2 = self._ccw(A, B, D)
        ccw3 = self._ccw(C, D, A)
        ccw4 = self._ccw(C, D, B)

        return ((ccw1 > 0 and ccw2 < 0) or (ccw1 < 0 and ccw2 > 0)) and \
               ((ccw3 > 0 and ccw4 < 0) or (ccw3 < 0 and ccw4 > 0))

    def update(
        self,
        tracks: Dict[int, Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        frame_idx: int = 0,
        fps: float = 25.0,
    ) -> Set[int]:
        """
        Update vehicle counts based on active tracked trajectories.

        :param tracks: Dictionary of active tracks from CentroidTracker.
        :param frame_width: Width of video frame.
        :param frame_height: Height of video frame.
        :param frame_idx: Current frame index.
        :param fps: Video FPS.
        :return: Set of object_ids that crossed the line in the current frame.
        """
        p1, p2 = self.get_absolute_line(frame_width, frame_height)
        newly_crossed = set()

        for obj_id, track in tracks.items():
            if obj_id in self.counted_ids:
                continue

            history = track["history"]
            if len(history) < 2:
                continue

            # Check segment between previous centroid and current centroid
            prev_pt = history[-2]
            curr_pt = history[-1]

            if self._intersect(p1, p2, prev_pt, curr_pt):
                # Determine movement direction based on Y movement relative to line
                # Downward movement (cy increases) vs Upward movement (cy decreases)
                dy = curr_pt[1] - prev_pt[1]
                move_dir = "down" if dy >= 0 else "up"

                if self.direction != "both" and move_dir != self.direction:
                    continue

                self.counted_ids.add(obj_id)
                newly_crossed.add(obj_id)
                self.total_count += 1
                self.counts_by_direction[move_dir] = self.counts_by_direction.get(move_dir, 0) + 1

                label = track.get("label", "Vehicle")
                self.counts_by_class[label] = self.counts_by_class.get(label, 0) + 1

                timestamp = frame_idx / fps if fps > 0 else 0.0
                self.recent_crossings.append({
                    "object_id": obj_id,
                    "frame_idx": frame_idx,
                    "timestamp_s": round(timestamp, 2),
                    "direction": move_dir,
                    "class_label": label,
                })

        return newly_crossed

    def reset(self) -> None:
        """Reset counter state."""
        self.counted_ids.clear()
        self.total_count = 0
        self.counts_by_direction = {"down": 0, "up": 0}
        self.counts_by_class.clear()
        self.recent_crossings.clear()
