"""
Vehicle Tracker Module
Implements multi-object tracking (MOT) using centroid distance matching,
trajectory history tracking, and disappearance management.
Aligned with CSE3010 Syllabus Module 4 (Object Tracking & Motion Estimation).
"""

import numpy as np
from collections import OrderedDict, deque
from scipy.spatial import distance as dist
from typing import Dict, List, Tuple, Any, Optional

# Tracked object representation:
# object_id -> { 'centroid': (cx, cy), 'bbox': (x, y, w, h), 'label': str, 'confidence': float, 'history': [(cx, cy)...] }


class CentroidTracker:
    """
    Tracks detected vehicles across consecutive frames by minimizing Euclidean distances
    between object centroids and updating track histories.
    """

    def __init__(self, max_disappeared: int = 25, max_distance: int = 60, trail_length: int = 20):
        """
        Initialize tracker.

        :param max_disappeared: Maximum consecutive frames an object can be missing before deregistering.
        :param max_distance: Maximum pixel distance to associate a detection with an existing track.
        :param trail_length: Maximum number of trajectory points retained for visualization and motion analysis.
        """
        self.next_object_id = 1
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.bboxes: OrderedDict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.labels: OrderedDict[int, str] = OrderedDict()
        self.confidences: OrderedDict[int, float] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.histories: OrderedDict[int, deque] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_length = trail_length

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], label: str, confidence: float) -> int:
        """Register a new detected object with a persistent unique ID."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.labels[obj_id] = label
        self.confidences[obj_id] = confidence
        self.disappeared[obj_id] = 0

        self.histories[obj_id] = deque(maxlen=self.trail_length)
        self.histories[obj_id].append(centroid)

        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int) -> None:
        """Deregister an object that has left the scene or was lost."""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.bboxes[object_id]
            del self.labels[object_id]
            del self.confidences[object_id]
            del self.disappeared[object_id]
            del self.histories[object_id]

    def update(self, detections: List[Tuple[int, int, int, int, float, str]]) -> Dict[int, Dict[str, Any]]:
        """
        Update tracking state with new frame detections.

        :param detections: List of (x, y, w, h, confidence, label) tuples.
        :return: Dictionary mapping active object_ids to object info.
        """
        # If no detections exist in current frame, increment disappearance counters
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_active_tracks()

        # Input detection centroids
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x, y, w, h, _, _) in enumerate(detections):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            input_centroids[i] = (cX, cY)

        # If currently tracking no objects, register all input detections
        if len(self.objects) == 0:
            for i, (x, y, w, h, conf, lbl) in enumerate(detections):
                self.register(tuple(input_centroids[i]), (x, y, w, h), lbl, conf)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute pairwise distance matrix between existing centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                new_centroid = tuple(input_centroids[col])
                bbox = (detections[col][0], detections[col][1], detections[col][2], detections[col][3])
                label = detections[col][5]
                confidence = detections[col][4]

                self.objects[object_id] = new_centroid
                self.bboxes[object_id] = bbox
                self.labels[object_id] = label
                self.confidences[object_id] = confidence
                self.disappeared[object_id] = 0
                self.histories[object_id].append(new_centroid)

                used_rows.add(row)
                used_cols.add(col)

            # Identify unused rows (disappeared existing objects)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Identify unused cols (new unmatched detections)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                bbox = (detections[col][0], detections[col][1], detections[col][2], detections[col][3])
                label = detections[col][5]
                confidence = detections[col][4]
                self.register(tuple(input_centroids[col]), bbox, label, confidence)

        return self._get_active_tracks()

    def _get_active_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Construct dictionary of all currently active tracks."""
        active = {}
        for obj_id, centroid in self.objects.items():
            active[obj_id] = {
                "centroid": centroid,
                "bbox": self.bboxes[obj_id],
                "label": self.labels[obj_id],
                "confidence": self.confidences[obj_id],
                "history": list(self.histories[obj_id]),
                "disappeared": self.disappeared[obj_id],
            }
        return active
