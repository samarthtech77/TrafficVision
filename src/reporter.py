"""
Reporter Module
Exports structured analytics reports:
- CSV log file (frame-by-frame stats)
- JSON summary metadata file
- Matplotlib analytics trend graphs (Count, Flow, Density over time)
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("TrafficVision.Reporter")


class TrafficReporter:
    """
    Collects execution analytics and generates CSV, JSON, and PNG chart reports.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def record_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        total_count: int,
        flow_vpm: float,
        active_vehicles: int,
        density_category: str,
        traffic_condition: str,
    ) -> None:
        """Add frame step statistics to collection."""
        self.records.append({
            "frame_idx": frame_idx,
            "timestamp_s": round(timestamp_s, 2),
            "cumulative_count": total_count,
            "flow_rate_vpm": round(flow_vpm, 2),
            "active_vehicles": active_vehicles,
            "density_category": density_category,
            "traffic_condition": traffic_condition,
        })

    def export_csv(self, filename: str = "traffic_report.csv") -> str:
        """Export frame step records to a CSV file."""
        csv_path = os.path.join(self.output_dir, filename)
        if not self.records:
            logger.warning("No records to export to CSV.")
            return csv_path

        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV report to '{csv_path}'")
        return csv_path

    def export_json(
        self,
        video_metadata: Dict[str, Any],
        counter_summary: Dict[str, Any],
        perf_stats: Dict[str, Any],
        filename: str = "traffic_report.json",
    ) -> str:
        """Export summary metadata and statistics to JSON file."""
        json_path = os.path.join(self.output_dir, filename)

        if self.records:
            df = pd.DataFrame(self.records)
            max_active = int(df["active_vehicles"].max())
            avg_flow = float(df["flow_rate_vpm"].mean())
            final_count = int(df["cumulative_count"].iloc[-1])
            final_condition = str(df["traffic_condition"].iloc[-1])
        else:
            max_active = 0
            avg_flow = 0.0
            final_count = counter_summary.get("total_count", 0)
            final_condition = "UNKNOWN"

        report_data = {
            "project": "TrafficVision: Automated Vehicle Tracking and Traffic Flow Analysis",
            "course": "CSE3010 - Computer Vision",
            "video_metadata": video_metadata,
            "performance": perf_stats,
            "summary_statistics": {
                "total_vehicles_counted": final_count,
                "peak_active_density": max_active,
                "average_flow_rate_vpm": round(avg_flow, 2),
                "final_traffic_condition": final_condition,
                "directional_counts": counter_summary.get("counts_by_direction", {}),
                "class_counts": counter_summary.get("counts_by_class", {}),
            },
            "recent_crossings": counter_summary.get("recent_crossings", [])[:50],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Saved JSON report to '{json_path}'")
        return json_path

    def export_plots(self, filename: str = "traffic_analytics.png") -> str:
        """Generates multi-panel Matplotlib analytics trend graph."""
        plot_path = os.path.join(self.output_dir, filename)
        if not self.records:
            logger.warning("No records available to generate plots.")
            return plot_path

        df = pd.DataFrame(self.records)
        time = df["timestamp_s"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("TrafficVision Analytics Summary Report", fontsize=14, fontweight="bold")

        # Subplot 1: Cumulative Vehicle Count
        ax1.plot(time, df["cumulative_count"], color="#007acc", linewidth=2.5, label="Cumulative Count")
        ax1.set_ylabel("Total Vehicles")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left")

        # Subplot 2: Traffic Flow Rate (veh/min)
        ax2.plot(time, df["flow_rate_vpm"], color="#2ca02c", linewidth=2.0, label="Flow Rate (veh/min)")
        ax2.set_ylabel("Flow (veh/min)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

        # Subplot 3: Active Vehicle Density
        ax3.plot(time, df["active_vehicles"], color="#d62728", linewidth=2.0, label="Active Density (vehicles)")
        ax3.axhline(y=3, color="green", linestyle=":", label="Low Threshold (<=3)")
        ax3.axhline(y=8, color="orange", linestyle=":", label="Moderate Threshold (<=8)")
        ax3.set_xlabel("Video Time (seconds)")
        ax3.set_ylabel("Active Vehicles")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)

        logger.info(f"Saved analytics plot to '{plot_path}'")
        return plot_path
