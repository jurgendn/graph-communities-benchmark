"""
Visualization utilities - simplified data loading and formatting helpers.
Lazy imports to avoid requiring numpy/matplotlib at import time.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.visualization.config import ConfigManager


def load_metric_file(path: Path) -> Dict:
    """Load metric data from JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_run(points: List[dict], method: str = "last") -> Optional[float]:
    """
    Summarize a run's points: last, max, or mean value.

    Args:
        points: List of {step, value} dicts
        method: 'last' (default), 'max', or 'mean'

    Returns:
        Float value or None if no valid data
    """
    vals: List[float] = []
    for p in points:
        if p and p.get("value") is not None:
            try:
                val = p.get("value")
                if val is not None:
                    val_float = float(val)
                    if val_float == val_float:  # Not NaN
                        vals.append(val_float)
            except (TypeError, ValueError):
                pass
    if not vals:
        return None

    if method == "max":
        return float(max(vals))
    elif method == "mean":
        return float(sum(vals) / len(vals))
    else:  # last
        return float(vals[-1])


def aggregate_series(runs: List[List[dict]]) -> Dict[float, List[float]]:
    """
    Aggregate multiple runs' time series by step.
    Returns dict mapping step -> [values at that step].
    """
    series: Dict[float, List[float]] = {}
    for run in runs:
        for idx, point in enumerate(run):
            step = point.get("step", float(idx))
            val = point.get("value")

            # Skip None/NaN values
            if val is None:
                continue
            try:
                val_float = float(val)
                # Skip NaN (need to import math for this check without numpy)
                if val_float != val_float:  # NaN check
                    continue
            except (TypeError, ValueError):
                continue

            try:
                step = float(step)
                series.setdefault(step, []).append(val_float)
            except (TypeError, ValueError):
                pass

    return series


# Configuration accessors (cached)
_config_cache = None

def _get_config() -> Dict:
    """Get config (cached)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = ConfigManager().get()
    return _config_cache


def get_plotter_config() -> Dict:
    """Get plotter config section."""
    return _get_config().get("plotter", {})


def get_algorithm_info() -> Dict:
    """Get algorithm styling info (names, colors, markers, order)."""
    cfg = get_plotter_config()
    common = cfg.get("common_plotter_settings", {})
    return {
        "names": common.get("methods_name", {}),
        "colors": common.get("colors", {}),
        "markers": common.get("markers", {}),
        "order": common.get("orders", []),
        "selected": common.get("selected_algorithms", []),
    }


def algorithm_display_name(alg: str) -> str:
    """Get display name for algorithm."""
    info = get_algorithm_info()
    return info.get("names", {}).get(alg, alg)


def algorithm_color(alg: str) -> Optional[str]:
    """Get color for algorithm."""
    info = get_algorithm_info()
    return info.get("colors", {}).get(alg)


def algorithm_marker(alg: str) -> Optional[str]:
    """Get marker for algorithm."""
    info = get_algorithm_info()
    return info.get("markers", {}).get(alg)


def get_dataset_info() -> Dict:
    """Get synthetic/real-world dataset info."""
    cfg = get_plotter_config()
    common = cfg.get("common_plotter_settings", {})
    return {
        "synthetic": common.get("synthetic_datasets", []),
        "real_world": common.get("real_world_datasets", []),
        "name_mapping": common.get("projects_name_mapping", {}),
    }


def project_display_name(project: str) -> str:
    """Get display name for project."""
    info = get_dataset_info()
    return info.get("name_mapping", {}).get(project, project)


def get_selected_algorithms() -> List[str]:
    """Get selected algorithms from config."""
    return get_algorithm_info().get("selected", [])


def sort_and_filter_algorithms(algs: List[str]) -> List[str]:
    """Sort algorithms by config order and filter to selected ones."""
    info = get_algorithm_info()
    order = info.get("order", [])
    selected = info.get("selected", [])

    # Filter to selected
    if selected:
        algs = [a for a in algs if a in selected]

    # Sort by order, then append rest
    result = [a for a in order if a in algs]
    result.extend([a for a in sorted(algs) if a not in result])
    return result
