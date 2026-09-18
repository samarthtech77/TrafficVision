"""
Synthetic Traffic Video Generator Utility
Generates a realistic multi-lane traffic video for testing and demonstration.
Ensures the TrafficVision pipeline can be executed immediately out-of-the-box.
"""

import os
import cv2
import numpy as np
import argparse


def generate_traffic_video(
    output_path: str = "videos/sample_traffic.mp4",
    width: int = 800,
    height: int = 480,
    fps: int = 25,
    duration_sec: int = 15,
) -> str:
    """
    Generates a synthetic multi-lane traffic video.

    :param output_path: File path to save output video.
    :param width: Frame width.
    :param height: Frame height.
    :param fps: Frames per second.
    :param duration_sec: Video length in seconds.
    :return: Path to created video file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_path = output_path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * duration_sec

    # Define moving vehicles with initial positions, velocities, colors, and dimensions
    vehicles = [
        # Lane 1 (Downwards)
        {"x": 180, "y": -60, "vy": 4.5, "vx": 0, "w": 45, "h": 70, "color": (50, 50, 220), "type": "Car"},
        {"x": 220, "y": -250, "vy": 5.0, "vx": 0, "w": 40, "h": 65, "color": (200, 100, 50), "type": "Car"},
        {"x": 190, "y": -480, "vy": 3.8, "vx": 0, "w": 55, "h": 110, "color": (80, 180, 80), "type": "Bus/Truck"},
        # Lane 2 (Downwards)
        {"x": 350, "y": -120, "vy": 6.2, "vx": 0, "w": 42, "h": 68, "color": (220, 220, 50), "type": "Car"},
        {"x": 360, "y": -350, "vy": 4.8, "vx": 0, "w": 25, "h": 45, "color": (50, 200, 200), "type": "Motorcycle"},
        {"x": 340, "y": -600, "vy": 5.5, "vx": 0, "w": 48, "h": 75, "color": (180, 50, 180), "type": "Car"},
        # Lane 3 (Upwards)
        {"x": 520, "y": height + 60, "vy": -5.2, "vx": 0, "w": 46, "h": 72, "color": (220, 120, 50), "type": "Car"},
        {"x": 540, "y": height + 300, "vy": -4.2, "vx": 0, "w": 52, "h": 95, "color": (60, 60, 60), "type": "Bus/Truck"},
        # Lane 4 (Upwards)
        {"x": 660, "y": height + 150, "vy": -6.0, "vx": 0, "w": 40, "h": 65, "color": (50, 180, 220), "type": "Car"},
    ]

    for frame_idx in range(total_frames):
        # 1. Background Road Scene
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Grass/Sides
        frame[:, :80] = (40, 80, 40)
        frame[:, width - 80:] = (40, 80, 40)

        # Road surface
        frame[:, 80:width - 80] = (50, 50, 50)

        # Road shoulder lines (Solid White)
        cv2.line(frame, (85, 0), (85, height), (240, 240, 240), 3)
        cv2.line(frame, (width - 85, 0), (width - 85, height), (240, 240, 240), 3)

        # Center Double Yellow Line
        mid_x = width // 2
        cv2.line(frame, (mid_x - 3, 0), (mid_x - 3, height), (0, 200, 255), 2)
        cv2.line(frame, (mid_x + 3, 0), (mid_x + 3, height), (0, 200, 255), 2)

        # Dashed Lane Lines
        dash_offset = (frame_idx * 4) % 40
        for lx in [270, width - 270]:
            for y_pos in range(-40 + dash_offset, height, 40):
                cv2.line(frame, (lx, y_pos), (lx, y_pos + 20), (220, 220, 220), 2)

        # 2. Draw Moving Vehicles
        for v in vehicles:
            v["y"] += v["vy"]

            # Loop vehicles when off screen
            if v["vy"] > 0 and v["y"] > height + 100:
                v["y"] = -100 - np.random.randint(20, 150)
            elif v["vy"] < 0 and v["y"] < -100:
                v["y"] = height + 100 + np.random.randint(20, 150)

            vx, vy = int(v["x"]), int(v["y"])
            vw, vh = v["w"], v["h"]

            # Draw vehicle body (if visible on frame)
            if -vh < vy < height + vh:
                # Vehicle body box
                cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), v["color"], -1)
                # Border
                cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), (10, 10, 10), 2)

                # Windshield & Headlights detail
                ws_y = vy - vh // 4 if v["vy"] > 0 else vy + vh // 4
                cv2.rectangle(frame, (vx - vw // 2 + 4, ws_y - 5), (vx + vw // 2 - 4, ws_y + 5), (180, 220, 240), -1)

                # Lights
                light_y = vy + vh // 2 - 3 if v["vy"] > 0 else vy - vh // 2 + 3
                light_clr = (0, 255, 255) if v["vy"] > 0 else (0, 0, 255)
                cv2.circle(frame, (vx - vw // 2 + 6, light_y), 3, light_clr, -1)
                cv2.circle(frame, (vx + vw // 2 - 6, light_y), 3, light_clr, -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic traffic video at: '{output_path}' ({width}x{height} @ {fps}fps, {duration_sec}s)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Traffic Video Generator")
    parser.add_argument("--output", type=str, default="videos/sample_traffic.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    args = parser.parse_args()

    generate_traffic_video(output_path=args.output, duration_sec=args.duration, fps=args.fps)
