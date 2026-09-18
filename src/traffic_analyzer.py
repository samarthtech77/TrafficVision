"""
Traffic Analyzer Module
Computes Traffic Flow (Vehicles per Minute), Traffic Density (Active Vehicles / ROI Occupancy),
and determines Traffic Condition state (LOW, MODERATE, HIGH/CONGESTED).
Aligned with CSE3010 Syllabus Module 4 (Spatio-Temporal Traffic Analysis).
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple


class TrafficAnalyzer:
    """
    Analyzes traffic parameters over time to calculate flow rates,
    density indicators, and classify traffic conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("traffic_analysis", {}) if config else {}

        dens_cfg = cfg.get("density_thresholds", {})
        self.low_density_thresh = dens_cfg.get("low", 3)
        self.mod_density_thresh = dens_cfg.get("moderate", 8)

        flow_cfg = cfg.get("flow_thresholds", {})
        self.low_flow_thresh = flow_cfg.get("low_vpm", 15.0)
        self.mod_flow_thresh = flow_cfg.get("moderate_vpm", 45.0)

        labels = cfg.get("condition_labels", {})
        self.lbl_low = labels.get("low", "LOW")
        self.lbl_mod = labels.get("moderate", "MODERATE")
        self.lbl_high = labels.get("high", "HIGH / CONGESTED")

        # Rolling window history for smooth metric computation
        self.window_size = 75  # ~3 seconds at 25 fps
        self.active_density_history = deque(maxlen=self.window_size)
        self.flow_history = deque(maxlen=self.window_size)

    def analyze(
        self,
        total_vehicles_crossed: int,
        active_vehicles_count: int,
        elapsed_seconds: float,
    ) -> Dict[str, Any]:
        """
        Computes traffic metrics for the current frame step.

        :param total_vehicles_crossed: Cumulative count of vehicles crossed.
        :param active_vehicles_count: Current count of active tracked vehicles in frame.
        :param elapsed_seconds: Total elapsed video time in seconds.
        :return: Dictionary containing flow, density, condition, and status metadata.
        """
        # 1. Traffic Flow Rate calculation (N vehicles / time in minutes)
        time_in_minutes = max(elapsed_seconds / 60.0, 0.05)  # Avoid division by zero
        flow_rate_vpm = round(total_vehicles_crossed / time_in_minutes, 2)

        # Update rolling density buffer
        self.active_density_history.append(active_vehicles_count)
        self.flow_history.append(flow_rate_vpm)

        avg_density = round(float(sum(self.active_density_history)) / len(self.active_density_history), 2)
        avg_flow = round(float(sum(self.flow_history)) / len(self.flow_history), 2)

        # 2. Traffic Density level determination
        if avg_density <= self.low_density_thresh:
            density_category = "LOW"
        elif avg_density <= self.mod_density_thresh:
            density_category = "MODERATE"
        else:
            density_category = "HIGH"

        # 3. Overall Traffic Condition classification matrix
        # Transparent rule-based classification based on active density and flow rate
        if avg_density <= self.low_density_thresh:
            condition = self.lbl_low
            condition_color = (0, 255, 0)  # Green
        elif avg_density <= self.mod_density_thresh:
            condition = self.lbl_mod
            condition_color = (0, 215, 255)  # Yellow / Gold
        else:
            condition = self.lbl_high
            condition_color = (0, 0, 255)  # Red / Congested

        return {
            "total_count": total_vehicles_crossed,
            "active_vehicles": active_vehicles_count,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "flow_rate_vpm": flow_rate_vpm,
            "avg_flow_rate_vpm": avg_flow,
            "instant_density": active_vehicles_count,
            "avg_density": avg_density,
            "density_category": density_category,
            "traffic_condition": condition,
            "condition_color": condition_color,
        }

    @staticmethod
    def classify_condition_static(
        active_count: int, low_thresh: int = 3, mod_thresh: int = 8
    ) -> str:
        """Helper method for standalone condition classification."""
        if active_count <= low_thresh:
            return "LOW"
        elif active_count <= mod_thresh:
            return "MODERATE"
        else:
            return "HIGH / CONGESTED"
