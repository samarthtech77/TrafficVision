"""
Unit tests for TrafficAnalyzer module.
"""

import pytest
from src.traffic_analyzer import TrafficAnalyzer


def test_traffic_analyzer_conditions():
    config = {
        "traffic_analysis": {
            "density_thresholds": {"low": 3, "moderate": 8},
            "flow_thresholds": {"low_vpm": 15.0, "moderate_vpm": 45.0},
        }
    }
    analyzer = TrafficAnalyzer(config)

    # Test LOW traffic condition
    stats_low = analyzer.analyze(total_vehicles_crossed=5, active_vehicles_count=2, elapsed_seconds=60.0)
    assert stats_low["traffic_condition"] == "LOW"
    assert stats_low["density_category"] == "LOW"
    assert stats_low["flow_rate_vpm"] == 5.0

    # Test MODERATE traffic condition
    stats_mod = analyzer.analyze(total_vehicles_crossed=25, active_vehicles_count=6, elapsed_seconds=60.0)
    assert stats_mod["density_category"] in ("LOW", "MODERATE")

    # Test HIGH / CONGESTED traffic condition with fresh analyzer
    analyzer_high = TrafficAnalyzer(config)
    stats_high = analyzer_high.analyze(total_vehicles_crossed=50, active_vehicles_count=12, elapsed_seconds=60.0)
    assert stats_high["traffic_condition"] == "HIGH / CONGESTED"


def test_classify_condition_static():
    assert TrafficAnalyzer.classify_condition_static(2, 3, 8) == "LOW"
    assert TrafficAnalyzer.classify_condition_static(5, 3, 8) == "MODERATE"
    assert TrafficAnalyzer.classify_condition_static(10, 3, 8) == "HIGH / CONGESTED"
