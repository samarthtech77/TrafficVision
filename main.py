"""
TrafficVision: Automated Vehicle Tracking and Traffic Flow Analysis
Master CLI Entry Point (CSE3010 - Computer Vision BYOP)
"""

import os
import sys
import time
import yaml
import argparse
import logging
import numpy as np
import cv2

# Local src modules
from src.video_processor import VideoProcessor
from src.preprocessor import ImagePreprocessor
from src.detector import get_detector
from src.tracker import CentroidTracker
from src.vehicle_counter import VehicleCounter
from src.motion_analyzer import MotionAnalyzer
from src.traffic_analyzer import TrafficAnalyzer
from src.visualizer import TrafficVisualizer
from src.reporter import TrafficReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficVision")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file or return defaults if missing."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config file '{config_path}' not found. Operating with default parameters.")
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="TrafficVision: Automated Vehicle Tracking & Traffic Flow Analysis"
    )
    parser.add_argument("--input", "-i", type=str, default="videos/sample_traffic.mp4", help="Input video file path")
    parser.add_argument("--output", "-o", type=str, default="results", help="Output directory path")
    parser.add_argument("--detector", "-d", type=str, default="bg_subtractor", choices=["bg_subtractor", "cascade"], help="Detector model")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml", help="Configuration YAML file")
    parser.add_argument("--no-video", action="store_true", help="Disable output annotated video generation")
    parser.add_argument("--no-plot", action="store_true", help="Disable output chart generation")

    args = parser.parse_args()

    # If default sample video is requested but does not exist, generate it automatically
    if args.input == "videos/sample_traffic.mp4" and not os.path.exists(args.input):
        print("Default sample video not found. Generating synthetic traffic video...")
        try:
            from generate_sample_video import generate_traffic_video
            generate_traffic_video(args.input)
        except Exception as e:
            logger.error(f"Failed to generate synthetic sample video: {e}")

    print("\nLoading video...")
    config = load_config(args.config)

    # 1. Initialize Video Processor
    try:
        vp = VideoProcessor(args.input, target_size=(
            config.get("video", {}).get("target_width", 800),
            config.get("video", {}).get("target_height", 480)
        ))
    except Exception as err:
        print(f"\n[ERROR] Video loading failed: {err}")
        sys.exit(1)

    print("Video loaded successfully.")
    print(f"Details: {vp.width}x{vp.height} @ {vp.fps:.2f} FPS | Total Frames: {vp.total_frames}")

    # 2. Instantiate pipeline components
    preprocessor = ImagePreprocessor(config)
    detector = get_detector(args.detector, config)
    tracker_cfg = config.get("tracker", {})
    tracker = CentroidTracker(
        max_disappeared=tracker_cfg.get("max_disappeared", 25),
        max_distance=tracker_cfg.get("max_distance", 60),
        trail_length=tracker_cfg.get("trail_length", 20),
    )

    counter_cfg = config.get("counter", {})
    counter = VehicleCounter(
        line_coords=counter_cfg.get("counting_line", ((0.05, 0.55), (0.95, 0.55))),
        direction=counter_cfg.get("counting_direction", "both"),
    )

    motion_analyzer = MotionAnalyzer(config)
    traffic_analyzer = TrafficAnalyzer(config)
    visualizer = TrafficVisualizer(config)
    reporter = TrafficReporter(args.output)

    # 3. Setup Video Writer
    writer = None
    output_video_path = ""
    if not args.no_video:
        out_filename = f"annotated_{os.path.basename(args.input)}"
        if not out_filename.endswith(".mp4") and not out_filename.endswith(".avi"):
            out_filename += ".mp4"
        output_video_path = os.path.join(args.output, out_filename)
        out_w, out_h = vp.get_output_dimensions()
        writer = VideoProcessor.create_writer(output_video_path, vp.fps, out_w, out_h)

    print("\nProcessing traffic video...")
    print("Vehicle detection started...")
    print("Vehicle tracking started...")
    print("Traffic analysis started...")

    start_time = time.time()
    processed_count = 0

    p1_line, p2_line = counter.get_absolute_line(*vp.get_output_dimensions())

    try:
        for frame_idx, frame in vp.get_frames():
            # A. Preprocessing (Grayscale, CLAHE, Blur)
            preprocessed = preprocessor.process(frame)

            # B. Vehicle Detection (MOG2 background subtraction + contours)
            detections = detector.detect(frame, preprocessed)

            # C. Vehicle Tracking (Centroid matching + trajectory updates)
            tracks = tracker.update(detections)

            # D. Motion Analysis & Optical Flow (Lucas-Kanade)
            motion_stats = motion_analyzer.process_frame(frame, tracks, fps=vp.fps)
            flow_vectors = motion_analyzer.get_flow_vectors()

            # E. Vehicle Counting (Line crossing orientation check)
            newly_crossed = counter.update(
                tracks, vp.get_output_dimensions()[0], vp.get_output_dimensions()[1], frame_idx, vp.fps
            )

            # F. Traffic Flow & Density Analysis
            elapsed_sec = (frame_idx + 1) / vp.fps
            traffic_stats = traffic_analyzer.analyze(
                total_vehicles_crossed=counter.total_count,
                active_vehicles_count=len(tracks),
                elapsed_seconds=elapsed_sec,
            )

            # G. Visual Annotation & Frame Rendering
            annotated_frame = visualizer.draw_annotation(
                frame=frame,
                tracks=tracks,
                motion_stats=motion_stats,
                traffic_stats=traffic_stats,
                counting_line=(p1_line, p2_line),
                flow_vectors=flow_vectors,
                newly_crossed_ids=newly_crossed,
            )

            # Save annotated frame to video writer
            if writer is not None:
                writer.write(annotated_frame)

            # Record metrics to reporter
            reporter.record_frame(
                frame_idx=frame_idx,
                timestamp_s=elapsed_sec,
                total_count=counter.total_count,
                flow_vpm=traffic_stats["flow_rate_vpm"],
                active_vehicles=traffic_stats["active_vehicles"],
                density_category=traffic_stats["density_category"],
                traffic_condition=traffic_stats["traffic_condition"],
            )

            processed_count += 1

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
    finally:
        vp.release()
        if writer is not None:
            writer.release()

    total_processing_time = time.time() - start_time
    proc_fps = processed_count / total_processing_time if total_processing_time > 0 else 0.0

    # 4. Export Reports
    csv_file = reporter.export_csv()
    json_file = reporter.export_json(
        video_metadata={
            "file_name": os.path.basename(args.input),
            "resolution": f"{vp.width}x{vp.height}",
            "fps": vp.fps,
            "total_frames": vp.total_frames,
            "duration_sec": round(vp.duration, 2),
        },
        counter_summary={
            "total_count": counter.total_count,
            "counts_by_direction": counter.counts_by_direction,
            "counts_by_class": counter.counts_by_class,
            "recent_crossings": counter.recent_crossings,
        },
        perf_stats={
            "processed_frames": processed_count,
            "total_processing_time_sec": round(total_processing_time, 2),
            "processing_fps": round(proc_fps, 2),
        },
    )

    plot_file = ""
    if not args.no_plot:
        plot_file = reporter.export_plots()

    # 5. Output Summary to Terminal
    final_stats = traffic_analyzer.analyze(counter.total_count, len(tracker.objects), vp.duration)

    print("\nAnalysis complete.\n")
    print(f"Total vehicles : {counter.total_count}")
    print(f"Vehicle flow   : {final_stats['flow_rate_vpm']:.1f} vehicles/min")
    print(f"Density        : {final_stats['density_category']}")
    print(f"Condition      : {final_stats['traffic_condition']}")
    print(f"\nResults saved to:")
    print(f"{os.path.abspath(args.output)}/")
    if output_video_path and os.path.exists(output_video_path):
        print(f"  - Video  : {output_video_path}")
    print(f"  - CSV    : {csv_file}")
    print(f"  - JSON   : {json_file}")
    if plot_file and os.path.exists(plot_file):
        print(f"  - Graph  : {plot_file}")


if __name__ == "__main__":
    main()
