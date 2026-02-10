"""
Data pipeline for fetching and merging Comet ML experiments.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import comet_ml

from src.visualization.common import extract_algorithm_name, normalize_params, to_float


class Fetcher:
    """Fetch experiments from Comet ML and save raw metrics."""

    def __init__(self, api_key: str, workspace: Optional[str] = None):
        if not api_key:
            raise ValueError("API key required")
        self.api_key = api_key
        self.workspace = workspace
        self.api = comet_ml.API(api_key=api_key)

    def list_experiments(self, project: str):
        """List all experiments for a project."""
        return self.api.get_experiments(workspace=self.workspace, project_name=project)

    def fetch_project(self, project: str, metric_keys: List[str], out_dir: Path = Path("data/raw")):
        """
        Fetch all experiments for a project and save raw metrics.

        Args:
            project: Comet ML project name
            metric_keys: List of metric names to extract
            out_dir: Base directory to save raw data
        """
        project_dir = out_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiments = self.list_experiments(project)
        except Exception as e:
            print(f"  Failed to list experiments for {project}: {e}")
            return

        if not experiments:
            print(f"  No experiments found for {project}")
            return

        for i, exp in enumerate(experiments, start=1):
            try:
                # Extract parameters
                try:
                    params = exp.get_parameters_summary()
                except Exception:
                    params = getattr(exp, "parameters", {}) or {}

                # Extract metrics
                try:
                    metrics = exp.get_metrics()
                except Exception:
                    metrics = getattr(exp, "metrics", []) or []

                # Filter to requested metrics only
                filtered = [m for m in metrics if m.get("metricName") in metric_keys]

                # Save raw data
                filename = project_dir / f"experiment_{i}.json"
                with open(filename, "w", encoding="utf-8") as fh:
                    json.dump({"parameters": params, "metrics": filtered}, fh, indent=2)

                print(f"  Saved raw {filename} (metrics={len(filtered)})")
            except Exception as e:
                print(f"  Failed processing experiment {i} in {project}: {e}")


class Merger:
    """Merge raw experimental data into aggregated metrics by algorithm."""

    def __init__(self, raw_dir: Path = Path("data/raw"), merge_dir: Path = Path("data/merge")):
        self.raw_dir = Path(raw_dir)
        self.merge_dir = Path(merge_dir)

    def merge_project(self, project: str, metric_keys: List[str]) -> bool:
        """
        Merge all raw experiments for a project into aggregated metrics.

        Args:
            project: Project name
            metric_keys: List of metric names to aggregate

        Returns:
            True if successful, False otherwise
        """
        project_raw = self.raw_dir / project
        project_merge = self.merge_dir / project

        if not project_raw.exists() or not project_raw.is_dir():
            print(f"  No raw data found for {project}")
            return False

        project_merge.mkdir(parents=True, exist_ok=True)

        # Initialize merged structure: {metric_name: {algorithm: [run1_points, run2_points, ...]}}
        merged_all: Dict[str, Dict[str, List[List[Dict[str, Any]]]]] = {k: {} for k in metric_keys}

        # Process each raw experiment file
        for i, raw_file in enumerate(sorted(project_raw.glob("*.json")), start=1):
            try:
                with open(raw_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                params = data.get("parameters", {})
                metrics = data.get("metrics", [])

                # Extract algorithm name
                norm_params = normalize_params(params)
                alg_name = extract_algorithm_name(norm_params, i)

                # Process each metric
                for metric_key in metric_keys:
                    metric_points = [m for m in metrics if m.get("metricName") == metric_key]
                    run_points = self._extract_points(metric_points)

                    if run_points:  # Only store if we have data
                        merged_all[metric_key].setdefault(alg_name, []).append(run_points)

            except Exception as e:
                print(f"  Failed processing raw file {raw_file}: {e}")

        # Write merged metrics
        success = True
        for metric_key in metric_keys:
            try:
                merged_metric = merged_all.get(metric_key, {})
                out_file = project_merge / f"{metric_key}.json"
                with open(out_file, "w", encoding="utf-8") as fh:
                    json.dump(merged_metric, fh, indent=2)
                print(f"  Wrote merged metric {metric_key} to {out_file}")
            except Exception as e:
                print(f"  Failed to write merged metric {metric_key} for {project}: {e}")
                success = False

        return success

    @staticmethod
    def _extract_points(metrics: List[dict]) -> List[Dict[str, Any]]:
        """
        Extract and normalize point data from metrics.
        Returns list of {step, value} dicts, sorted by step.
        """
        points: List[Dict[str, Any]] = []
        for m in metrics:
            # Extract step
            step = m.get("step")
            if step is None:
                step = m.get("epoch") or m.get("x")

            # Extract value
            val = None
            for key in ("metricValue", "value", "y"):
                if key in m:
                    val = m[key]
                    break

            # Convert to float
            num = to_float(val)
            if num is None:
                continue

            # Normalize step to int
            try:
                step_int: Optional[int] = int(step) if step is not None else None
            except (TypeError, ValueError):
                try:
                    step_int = int(float(step or 0))
                except (TypeError, ValueError):
                    step_int = None

            if step_int is None:
                step_int = len(points)  # Use index as fallback

            points.append({"step": step_int, "value": num})

        # Sort by step
        points.sort(key=lambda x: x["step"])
        return points
