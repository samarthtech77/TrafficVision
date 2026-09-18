"""
Unit tests for VehicleCounter module.
"""

import pytest
from src.vehicle_counter import VehicleCounter


def test_vehicle_counter_line_crossing():
    # Horizontal reference line at y = 200 (normalized y=0.5 on 400h frame)
    counter = VehicleCounter(line_coords=((0.0, 0.5), (1.0, 0.5)), direction="both")

    # Vehicle moving from top (y=150) to bottom (y=250) across y=200
    tracks = {
        1: {
            "centroid": (200, 250),
            "history": [(200, 150), (200, 250)],
            "label": "Vehicle",
        }
    }

    crossed = counter.update(tracks, frame_width=400, frame_height=400, frame_idx=1, fps=25.0)
    assert 1 in crossed
    assert counter.total_count == 1
    assert counter.counts_by_direction["down"] == 1

    # Second frame: duplicate count prevention
    crossed_again = counter.update(tracks, frame_width=400, frame_height=400, frame_idx=2, fps=25.0)
    assert len(crossed_again) == 0
    assert counter.total_count == 1  # Total count unchanged
