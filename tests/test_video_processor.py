"""
Unit tests for VideoProcessor module.
"""

import os
import pytest
import numpy as np
import cv2
from src.video_processor import VideoProcessor


@pytest.fixture
def temp_video_file(tmp_path):
    """Creates a temporary 10-frame dummy MP4 video file for testing."""
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))
    for i in range(10):
        frame = np.full((480, 640, 3), i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_video_processor_valid(temp_video_file):
    vp = VideoProcessor(temp_video_file)
    assert vp.width == 640
    assert vp.height == 480
    assert vp.total_frames == 10
    assert vp.fps == 25.0

    frames = list(vp.get_frames())
    assert len(frames) == 10
    assert frames[0][1].shape == (480, 640, 3)
    vp.release()


def test_video_processor_resize(temp_video_file):
    vp = VideoProcessor(temp_video_file, target_size=(320, 240))
    assert vp.get_output_dimensions() == (320, 240)

    frames = list(vp.get_frames())
    assert len(frames) == 10
    assert frames[0][1].shape == (240, 320, 3)
    vp.release()


def test_video_processor_missing_file():
    with pytest.raises(FileNotFoundError):
        VideoProcessor("non_existent_file.mp4")
