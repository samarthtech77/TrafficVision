"""
Video Processor Module
Handles video ingestion, path validation, frame extraction, metadata retrieval,
and video output writing.
"""

import os
import cv2
import logging
from typing import Generator, Tuple, Optional

logger = logging.getLogger("TrafficVision.VideoProcessor")


class VideoProcessor:
    """
    Handles video reading, validation, frame retrieval, and writing.
    """

    def __init__(self, video_path: str, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize VideoProcessor.

        :param video_path: Path to the input video file.
        :param target_size: Optional tuple (width, height) to resize frames.
        """
        self.video_path = video_path
        self.target_size = target_size
        self.cap = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.duration = 0.0

        self._validate_and_open()

    def _validate_and_open(self) -> None:
        """
        Validates the video file existence and opens it with OpenCV.
        Raises FileNotFoundError or ValueError if invalid.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at: '{self.video_path}'")

        if not os.path.isfile(self.video_path):
            raise ValueError(f"Path is not a valid file: '{self.video_path}'")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: '{self.video_path}'")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps > 240:
            logger.warning(f"Unusual FPS detected ({self.fps}), defaulting to 25.0 FPS.")
            self.fps = 25.0

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid frame dimensions ({self.width}x{self.height}) in video.")

        if self.total_frames > 0:
            self.duration = self.total_frames / self.fps
        else:
            self.duration = 0.0

        logger.info(
            f"Opened video '{self.video_path}': {self.width}x{self.height} @ {self.fps:.2f} FPS, "
            f"Total Frames: {self.total_frames}, Duration: {self.duration:.2f}s"
        )

    def get_output_dimensions(self) -> Tuple[int, int]:
        """Returns effective (width, height) after optional target resizing."""
        if self.target_size:
            return self.target_size
        return (self.width, self.height)

    def get_frames(self) -> Generator[Tuple[int, cv2.Mat], None, None]:
        """
        Yields (frame_index, frame_mat) sequentially.
        Resizes frame if target_size is set.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Video stream is not open.")

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            if self.target_size and (frame.shape[1] != self.target_size[0] or frame.shape[0] != self.target_size[1]):
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        """Release OpenCV video capture handle."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @staticmethod
    def create_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
        """
        Creates a cv2.VideoWriter for saving annotated video.

        :param output_path: Destination path (.mp4 or .avi).
        :param fps: Output video frame rate.
        :param width: Frame width.
        :param height: Frame height.
        :return: Initialized cv2.VideoWriter object.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()

        if ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning(f"Primary FourCC failed for {output_path}. Trying fallback 'XVID'...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        return writer
