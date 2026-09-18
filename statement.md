# Problem Statement and Project Scope

**Project Title:** TrafficVision: Automated Vehicle Tracking and Traffic Flow Analysis Using Computer Vision

---

## 1. Problem Statement

Traffic congestion and inefficient traffic monitoring are significant challenges in urban road management. Conventional traffic monitoring often relies on manual observation or basic vehicle counting sensors (such as inductive loops), which fail to provide continuous spatial information regarding vehicle trajectories, instant speeds, spatio-temporal flow rates, and density variations.

**TrafficVision** proposes an end-to-end Computer Vision-based system that processes pre-recorded traffic videos to automatically:

1. Detect vehicles within video frames using classical background modeling and morphological filtering.
2. Track vehicle identities across consecutive frames using Euclidean distance centroid association.
3. Count traffic participants crossing virtual reference gates without double counting.
4. Analyze vehicle spatio-temporal motion parameters using Lucas-Kanade (KLT) Optical Flow.
5. Estimate traffic flow rate ($N/T$ vehicles per minute) and active vehicle density per frame.
6. Classify overall traffic conditions into transparent categories (`LOW`, `MODERATE`, `HIGH / CONGESTED`).

---

## 2. Project Scope

### In-Scope Capabilities

- Processing pre-recorded traffic video streams (MP4, AVI, MOV).
- Grayscale preprocessing, Gaussian smoothing, and CLAHE contrast enhancement.
- Foreground-background segmentation using MOG2 (Mixture of Gaussians) background subtraction.
- Morphological noise removal (Opening, Closing, Dilation) and contour area/aspect ratio filtering.
- Multi-object centroid tracking with track history management and disappearance handling.
- Virtual counting line crossing detection using 2D vector cross-product orientation logic.
- Motion vector estimation and sparse Lucas-Kanade (KLT) Optical Flow analysis.
- Flow rate ($N/T$ veh/min), active density estimation, and condition classification matrix.
- Automated generation of annotated video output (`annotated_sample_traffic.mp4`), structured CSV logs (`traffic_report.csv`), JSON metadata summaries (`traffic_report.json`), and Matplotlib analytics charts (`traffic_analytics.png`).

### Out-of-Scope

- City-wide live CCTV streaming network integration.
- Hardware IoT microcontrollers or automatic physical traffic light control.
- Automatic Number Plate Recognition (ANPR) or driver face recognition.
- 3D vehicle reconstruction or multi-camera stereoscopic geometry calibration.
- Commercial production deployment or cloud server orchestration.

---

## 3. Target Users

1. **Traffic Engineers & Transportation Planning Authorities**: Analyzing traffic density patterns, bottleneck locations, and peak flow hours from surveillance footage.
2. **Researchers & Developers**: Analyzing and evaluating computer vision-based traffic detection, tracking, motion, and flow analysis methods.
3. **Smart City Analysts**: Reviewing historical traffic video datasets for road capacity assessment.

---

## 4. High-Level Features

- **Modular Computer Vision Engine**: Clean separation between video ingestion, preprocessing, detection, tracking, counting, motion analysis, traffic classification, visualization, and reporting.
- **Dual Detection Support**: Primary MOG2 classical CV detector with optional cascade model support.
- **Directional Line Crossing Counter**: Accurate counting with top-to-bottom and bottom-to-top directional breakdown.
- **HUD Dashboard Overlay**: Real-time stats header banner embedded directly into the generated output video.
- **CLI Workflows**: Fast, configurable command-line interface with customizable thresholds via `config/config.yaml`.