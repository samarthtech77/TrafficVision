# TrafficVision Sample Videos Directory

This directory contains input video files for the TrafficVision system.

## Generating Sample Synthetic Video

You can automatically generate a synthetic multi-lane traffic video for testing without downloading any external files:

```bash
python generate_sample_video.py --output videos/sample_traffic.mp4 --duration 15 --fps 25
```

## Supported Video Formats

- `.mp4` (Recommended, H.264 / AVC codec)
- `.avi` (XVID / MJPEG)
- `.mov`

## Recommended Video Specifications

- Resolution: 640x480 to 1920x1080
- Frame Rate: 20 FPS to 60 FPS
- Angle: Top-down, elevated overhead view, or high-angle perspective road surveillance.
