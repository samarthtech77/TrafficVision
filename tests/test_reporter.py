"""
Unit tests for TrafficReporter module.
"""

import os
import json
import pytest
import pandas as pd
from src.reporter import TrafficReporter


def test_reporter_exports(tmp_path):
    output_dir = str(tmp_path / "results")
    reporter = TrafficReporter(output_dir)

    reporter.record_frame(0, 0.0, 0, 0.0, 1, "LOW", "LOW")
    reporter.record_frame(25, 1.0, 2, 120.0, 3, "LOW", "LOW")

    csv_path = reporter.export_csv("test_report.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "cumulative_count" in df.columns

    json_path = reporter.export_json(
        video_metadata={"fps": 25},
        counter_summary={"total_count": 2},
        perf_stats={"processed_frames": 25},
        filename="test_report.json",
    )
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["summary_statistics"]["total_vehicles_counted"] == 2

    plot_path = reporter.export_plots("test_plot.png")
    assert os.path.exists(plot_path)
