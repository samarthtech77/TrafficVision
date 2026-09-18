# TrafficVision: Automated Vehicle Tracking and Traffic Flow Analysis Using Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![Testing](https://img.shields.io/badge/Tests-13%20Passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

## Project Overview

The system analyzes **pre-recorded traffic videos** and automatically extracts useful traffic information such as vehicle detections, tracking identities, vehicle crossings, motion vectors, traffic flow, active vehicle density, and overall traffic condition.

The project implements a complete classical Computer Vision pipeline using:

- Image preprocessing
- CLAHE contrast enhancement
- Gaussian filtering
- ROI masking
- MOG2 background subtraction
- Morphological image processing
- Contour-based vehicle detection
- Centroid-based multi-object tracking
- Virtual reference-line vehicle counting
- Lucas-Kanade (KLT) optical flow
- Motion and velocity estimation
- Traffic flow and density analysis
- Traffic condition classification
- Video visualization
- CSV/JSON reporting
- Analytics visualization

The system is designed as a lightweight, modular, command-line-executable traffic analysis pipeline.

---

## Problem Statement

Traffic congestion and inefficient traffic monitoring are significant challenges in urban transportation. Conventional traffic monitoring may rely on manual observation or basic vehicle-counting methods, which provide limited information about vehicle movement, trajectories, flow, and density over time.

TrafficVision addresses this problem by processing traffic video and automatically:

1. Detecting moving vehicles.
2. Tracking vehicles across consecutive frames.
3. Assigning persistent tracking IDs.
4. Detecting vehicle crossings over a virtual reference line.
5. Determining movement direction.
6. Analyzing vehicle motion using Lucas-Kanade optical flow.
7. Estimating traffic flow in vehicles per minute.
8. Estimating active vehicle density.
9. Classifying traffic conditions as `LOW`, `MODERATE`, or `HIGH / CONGESTED`.
10. Generating annotated video and structured traffic reports.

---

## Objectives

The main objectives of TrafficVision are:

- Develop a practical Computer Vision pipeline for traffic-video analysis.
- Apply image preprocessing and enhancement techniques to traffic frames.
- Detect moving vehicles using background subtraction and contour analysis.
- Track vehicles using persistent IDs across frames.
- Count vehicles crossing a virtual reference line.
- Determine vehicle movement direction.
- Analyze motion using sparse Lucas-Kanade optical flow.
- Estimate traffic flow and active traffic density.
- Classify the observed traffic condition.
- Generate visual and structured analytical outputs.
- Maintain a modular, testable and command-line-executable implementation.

---

# System Pipeline

```text
                 Input Traffic Video
                         │
                         ▼
              Video Validation & Reading
                         │
                         ▼
                Image Preprocessing
          ┌──────────────┼──────────────┐
          │              │              │
       Grayscale       CLAHE       Gaussian Blur
                         │
                         ▼
                     ROI Masking
                         │
                         ▼
              MOG2 Background Subtraction
                         │
                         ▼
              Morphological Processing
                         │
                         ▼
             Contour Detection & Filtering
                         │
                         ▼
                 Vehicle Detections
                         │
                         ▼
               Centroid-Based Tracking
                         │
                ┌────────┴────────┐
                ▼                 ▼
        Vehicle Counter      Motion Analyzer
        Virtual Line         KLT Optical Flow
        Crossing             Motion Vectors
                │                 │
                └────────┬────────┘
                         ▼
                Traffic Analyzer
             Flow + Density + Condition
                         │
                         ▼
                    Visualization
                         │
                         ▼
                     Reporting
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           Video        CSV         JSON
                         │
                         ▼
                   Analytics Plot
```

---

# Key Features

## 1. Video Ingestion and Validation

TrafficVision loads pre-recorded traffic videos and validates the input before processing.

The video processor extracts:

- Frame width
- Frame height
- FPS
- Total frame count
- Video duration

Supported formats depend on the codecs available through the local OpenCV installation and include common formats such as MP4, AVI and MOV.

---

## 2. Image Preprocessing

The preprocessing pipeline improves the quality and consistency of traffic frames before vehicle detection.

Implemented operations include:

- Grayscale conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian smoothing
- Region-of-Interest (ROI) masking

The preprocessing stage reduces unnecessary image information and improves the input used by the subsequent foreground-detection pipeline.

---

## 3. Foreground Vehicle Detection

TrafficVision uses classical Computer Vision techniques rather than requiring a large deep-learning model.

The detector uses:

- MOG2 background subtraction
- Morphological opening
- Morphological closing
- Dilation
- Contour extraction
- Contour-area filtering
- Bounding-box aspect-ratio filtering

The resulting contours are converted into candidate vehicle bounding boxes.

---

## 4. Multi-Object Vehicle Tracking

The tracking module maintains vehicle identities across consecutive frames.

The tracker:

- Assigns persistent tracking IDs.
- Associates detections between frames.
- Uses centroid-based Euclidean distance.
- Maintains track histories.
- Handles temporarily missing detections.
- Stores vehicle trajectories.

This allows individual vehicle movement to be analyzed over time rather than treating every frame independently.

---

## 5. Directional Vehicle Counting

TrafficVision uses a configurable virtual reference line to identify vehicle crossing events.

The system:

- Monitors vehicle trajectories.
- Detects when a trajectory crosses the reference line.
- Uses vector orientation/cross-product logic.
- Prevents repeated counting of the same track.
- Determines crossing direction.
- Maintains directional counts.

Example directions include:

```text
UP
DOWN
```

The counting-line parameters can be configured through:

```text
config/config.yaml
```

---

## 6. Motion Analysis

TrafficVision performs motion analysis using both vehicle trajectories and optical flow.

Implemented techniques include:

- Centroid displacement
- Track history
- Motion vectors
- Sparse Lucas-Kanade Optical Flow
- Image-space velocity estimation

The Lucas-Kanade implementation uses OpenCV's:

```python
cv2.calcOpticalFlowPyrLK()
```

The calculated velocity is an **image-space measurement** and should not be interpreted as calibrated real-world speed in km/h unless camera calibration and perspective transformation are introduced.

---

## 7. Traffic Flow Analysis

Traffic flow is estimated from the number of vehicles crossing the configured reference line.

The basic calculation is:

```text
Traffic Flow Rate =
Number of Vehicles Crossed / Elapsed Time in Minutes
```

The output unit is:

```text
vehicles/min
```

---

## 8. Traffic Density Analysis

Active traffic density is estimated from the number of currently tracked vehicles within the configured analysis region.

The system uses configurable thresholds to determine the corresponding traffic-density state.

---

## 9. Traffic Condition Classification

TrafficVision classifies traffic conditions into:

```text
LOW
MODERATE
HIGH / CONGESTED
```

The thresholds are configurable through:

```text
config/config.yaml
```

These thresholds are project-specific analysis parameters and should not be interpreted as universal traffic-engineering standards.

---

## 10. Visualization

The annotated output video can display:

- Vehicle bounding boxes
- Vehicle labels
- Persistent tracking IDs
- Motion/velocity information
- Vehicle trajectory trails
- Optical-flow vectors
- Virtual counting line
- Traffic statistics HUD

This provides visual verification of the Computer Vision pipeline.

---

## 11. Reporting

TrafficVision generates multiple output formats:

```text
Annotated MP4 Video
CSV Statistics Report
JSON Metadata Report
Analytics Plot
```

This allows the system's results to be inspected both visually and numerically.

---

## 12. Synthetic Traffic Video Generator

The repository contains:

```text
generate_sample_video.py
```

This script generates a self-contained synthetic traffic video for testing.

This allows the complete project to be executed without downloading an external traffic dataset.

---

# System Architecture

The repository contains the system architecture diagram at:

```text
docs/architecture.svg
```

The logical architecture is:

```text
                 ┌────────────────────────┐
                 │   Input Traffic Video  │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    Video Processor     │
                 │ Validation + Metadata  │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │     Preprocessor       │
                 │ Gray + CLAHE + Blur    │
                 │        + ROI            │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    Vehicle Detector    │
                 │ MOG2 + Morphology +    │
                 │      Contours          │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    Centroid Tracker    │
                 │ IDs + Track History    │
                 └────────────┬───────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Vehicle Counter │   │ Motion Analyzer │
          │ Line Crossing   │   │ KLT Optical Flow│
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                 ┌────────────────────────┐
                 │    Traffic Analyzer    │
                 │ Flow + Density +       │
                 │ Condition              │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │       Visualizer       │
                 │ HUD + Boxes + Trails   │
                 └────────────┬───────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Output Video    │   │ TrafficReporter │
          │ Annotated MP4   │   │ CSV + JSON +    │
          │                 │   │ Analytics Plot  │
          └─────────────────┘   └─────────────────┘
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Scientific Computing | SciPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Testing | PyTest |
| Background Modeling | MOG2 |
| Optical Flow | Lucas-Kanade |
| Object Tracking | Centroid-based Euclidean association |
| Video Processing | OpenCV VideoCapture / VideoWriter |

The complete dependency list is available in:

```text
requirements.txt
```

---

# Repository Structure

```text
TrafficVision/
│
├── README.md
├── statement.md
├── requirements.txt
├── main.py
├── generate_sample_video.py
├── LICENSE
├── .gitignore
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── preprocessor.py
│   ├── detector.py
│   ├── tracker.py
│   ├── vehicle_counter.py
│   ├── motion_analyzer.py
│   ├── traffic_analyzer.py
│   ├── visualizer.py
│   └── reporter.py
│
├── videos/
│   ├── README.md
│   └── sample_traffic.mp4
│
├── results/
│   ├── .gitkeep
│   ├── annotated_sample_traffic.mp4
│   ├── traffic_report.csv
│   ├── traffic_report.json
│   └── traffic_analytics.png
│
├── tests/
│   ├── __init__.py
│   ├── test_video_processor.py
│   ├── test_preprocessor.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_vehicle_counter.py
│   ├── test_traffic_analyzer.py
│   └── test_reporter.py
│
└── docs/
    ├── architecture.svg
    └── PROJECT_REPORT.md
```

---

# Requirements

## Software

Recommended environment:

- Python 3.8 or later
- pip
- Git
- Windows, Linux or macOS

## Python Dependencies

Install all required dependencies using:

```powershell
pip install -r requirements.txt
```

The project uses:

- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- PyYAML
- PyTest

---

# Local Installation and Setup

## Step 1 — Open the Project Directory

Open **PowerShell** or **Command Prompt**.

Navigate to the project directory.

---

## Step 2 — Create a Virtual Environment

A virtual environment is recommended but optional.

```powershell
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Or in Command Prompt:

```cmd
venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

Run:

```powershell
pip install -r requirements.txt
```

---

# Running the Project

TrafficVision provides a synthetic traffic video generator, allowing the complete system to be tested without downloading external data.

## Step 1 — Generate Sample Traffic Video

Run:

```powershell
python generate_sample_video.py --output videos/sample_traffic.mp4 --duration 15 --fps 25
```

This creates:

```text
videos/sample_traffic.mp4
```

The sample video contains:

```text
Resolution : 800 × 480
Frame Rate : 25 FPS
Duration   : 15 seconds
Frames     : 375
```

---

## Step 2 — Run the Main Pipeline

Run:

```powershell
python main.py --input videos/sample_traffic.mp4 --output results/
```

The system will:

1. Load and validate the video.
2. Read video metadata.
3. Preprocess frames.
4. Perform MOG2 background subtraction.
5. Apply morphological filtering.
6. Detect vehicle contours.
7. Track vehicles across frames.
8. Detect reference-line crossings.
9. Analyze vehicle motion.
10. Estimate traffic flow.
11. Estimate traffic density.
12. Classify traffic condition.
13. Generate annotated output video.
14. Generate CSV and JSON reports.
15. Generate the traffic analytics plot.

---

# Example Terminal Output

A successful execution follows this general structure:

```text
Loading video...

Video loaded successfully.

Details: 800x480 @ 25.00 FPS | Total Frames: 375

Processing traffic video...

Vehicle detection started...

Vehicle tracking started...

Traffic analysis started...

Analysis complete.

Total vehicles : <generated result>
Vehicle flow   : <generated result> vehicles/min
Density        : <generated result>
Condition      : <generated result>

Results saved to:
results/

  - Video  : results/annotated_sample_traffic.mp4
  - CSV    : results/traffic_report.csv
  - JSON   : results/traffic_report.json
  - Graph  : results/traffic_analytics.png
```

The numerical values are dependent on the input video and configuration. They should not be treated as fixed benchmark values.

---

# Checking the Generated Results

After execution, open:

```text
results/
```

## 1. Annotated Video

```text
results/annotated_sample_traffic.mp4
```

The annotated video can display:

- Vehicle bounding boxes
- Tracking IDs
- Vehicle labels
- Motion/velocity information
- Vehicle trajectory trails
- Optical-flow vectors
- Virtual counting line
- Traffic-analysis HUD

The video can be viewed using VLC, Windows Media Player or another compatible media player.

---

## 2. Analytics Plot

```text
results/traffic_analytics.png
```

The plot visualizes traffic statistics over time, including measurements such as:

- Vehicle count
- Flow rate
- Active vehicle density

---

## 3. CSV Report

```text
results/traffic_report.csv
```

The CSV contains frame-level traffic statistics such as:

- Frame index
- Vehicle counts
- Active tracked vehicles
- Flow information
- Density
- Traffic condition
- Relevant crossing/motion information

The file can be opened using Excel or another spreadsheet application.

---

## 4. JSON Report

```text
results/traffic_report.json
```

The JSON file contains structured execution and traffic-analysis information such as:

- Video metadata
- Analysis summary
- Directional counts
- Vehicle statistics
- Performance information
- Crossing events where available

---

# Using a Custom Traffic Video

TrafficVision can also process a custom pre-recorded traffic video.

## Step 1 — Add the Video

Place your video inside:

```text
videos/
```

For example:

```text
videos/my_traffic.mp4
```

## Step 2 — Run the Pipeline

```powershell
python main.py --input videos/my_traffic.mp4 --output results/
```

The generated outputs will be stored in:

```text
results/
```

## Recommended Video Conditions

For meaningful results, the input video should preferably:

- Clearly show moving vehicles.
- Have a relatively stable camera.
- Contain a visible road/traffic region.
- Have reasonable lighting.
- Match the configured ROI.
- Match the configured reference counting line.

If the camera viewpoint or road geometry changes significantly, update the appropriate parameters in:

```text
config/config.yaml
```

---

# Additional Command-Line Options

## Skip Video Rendering

For faster analysis when an annotated output video is not required:

```powershell
python main.py --input videos/sample_traffic.mp4 --no-video
```

## Specify a Custom Configuration File

```powershell
python main.py --input videos/sample_traffic.mp4 --config config/config.yaml
```

For all available command-line options:

```powershell
python main.py --help
```

---

# Configuration

TrafficVision uses:

```text
config/config.yaml
```

The configuration file controls important parameters for the different modules.

## Detector Parameters

Examples include:

- MOG2 history
- Variance threshold
- Minimum contour area
- Maximum contour area
- Bounding-box aspect-ratio constraints

## Tracker Parameters

Examples include:

- Maximum disappeared frames
- Maximum association distance
- Track-history length

## Vehicle Counter Parameters

Examples include:

- Reference-line endpoints
- Direction filter

## Traffic Analyzer Parameters

Examples include:

- Density thresholds
- Flow-rate thresholds
- Traffic-condition classification parameters

This configuration-based design allows the system to be adapted to different traffic scenes without modifying the core source code.

---

# Testing

TrafficVision includes an automated PyTest test suite.

Run:

```powershell
python -m pytest tests/
```

The tests cover:

- Video processor
- Preprocessor
- Detector
- Tracker
- Vehicle counter
- Traffic analyzer
- Reporter

## Verified Test Execution

The current test suite contains:

```text
13 tests
```

and all tests passed:

```text
13 passed
```

A successful test run should report all tests as passed.

```text
============================= test session starts =============================

collected 13 items

tests/test_detector.py              ..
tests/test_preprocessor.py          ..
tests/test_reporter.py              .
tests/test_tracker.py               ..
tests/test_traffic_analyzer.py      ..
tests/test_vehicle_counter.py       .
tests/test_video_processor.py       ...

============================= 13 passed =============================
```

If the source code, dependencies or configuration are changed, the test suite should be executed again.

---

# Verified Local Execution

The complete pipeline has been successfully executed using the included synthetic traffic video.

## Input Video

```text
Resolution : 800 × 480
Frame Rate : 25 FPS
Frames     : 375
Duration   : 15 seconds
```

## Generated Outputs

```text
results/annotated_sample_traffic.mp4
results/traffic_report.csv
results/traffic_report.json
results/traffic_analytics.png
```

## Automated Testing

```text
13 / 13 tests passed
```

The numerical traffic-analysis results should always be taken from the current execution because they depend on the input video and configuration.

---

# Project Limitations

TrafficVision is an academic prototype and has the following limitations:

1. It is primarily designed for pre-recorded traffic videos.
2. It is not a production-ready city-wide traffic management system.
3. MOG2 background subtraction can be affected by significant illumination changes, camera movement, heavy rain, fog, shadows and other dynamic-background conditions.
4. The current detector uses classical foreground/background segmentation and contour-based filtering rather than a deep-learning object detector.
5. Contour-based vehicle classification is less robust than learned object-classification approaches.
6. Image-space velocity measurements are not equivalent to calibrated real-world speed in km/h.
7. Real-world speed estimation would require camera calibration and perspective/homography transformation.
8. Traffic-condition thresholds are project-specific rather than universal traffic-engineering standards.
9. Performance depends on video resolution, scene complexity and available hardware.

---

# Future Enhancements

Possible future improvements include:

1. Integration of a lightweight object detector such as YOLOv8-Nano or another efficient model.
2. Improved vehicle-class recognition.
3. Improved tracking for crowded traffic scenes.
4. Perspective/homography calibration for real-world distance estimation.
5. Real-world speed estimation in km/h.
6. Adaptive handling of changing lighting and weather conditions.
7. Live CCTV/RTSP camera support.
8. Interactive dashboard for traffic analytics.
9. Evaluation using larger annotated traffic datasets.
10. Historical traffic modelling and congestion prediction.

These features are considered future enhancements and are not part of the current academic MVP unless explicitly implemented.

---

## Technical Implementation

The project includes:

- Modular source code
- Configuration management
- Multiple Python modules/classes
- Automated testing
- Command-line execution
- Input/output validation
- Documentation
- Result generation

# References

- OpenCV Documentation
- Python Documentation
- NumPy Documentation
- SciPy Documentation
- Pandas Documentation
- Matplotlib Documentation
- PyYAML Documentation
- PyTest Documentation

# License

This project is released under the [MIT License](LICENSE).

See [`LICENSE`](LICENSE) for details.

---
