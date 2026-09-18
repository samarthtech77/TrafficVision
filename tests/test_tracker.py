"""
Unit tests for CentroidTracker module.
"""

import pytest
from src.tracker import CentroidTracker


def test_tracker_register_and_update():
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    # Frame 1: One detection at (100, 100, 40, 40)
    det1 = [(100, 100, 40, 40, 0.9, "Vehicle")]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    obj_id = list(tracks1.keys())[0]
    assert tracks1[obj_id]["centroid"] == (120, 120)

    # Frame 2: Vehicle moves slightly to (105, 105, 40, 40)
    det2 = [(105, 105, 40, 40, 0.9, "Vehicle")]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert list(tracks2.keys())[0] == obj_id  # Persistent ID retained
    assert tracks2[obj_id]["centroid"] == (125, 125)
    assert len(tracks2[obj_id]["history"]) == 2


def test_tracker_deregistration():
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)
    det = [(100, 100, 40, 40, 0.9, "Vehicle")]
    tracker.update(det)

    # Disappears for 3 frames
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])

    assert len(tracks) == 0  # Deregistered
