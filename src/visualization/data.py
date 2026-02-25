"""
Data pipeline for fetching and merging Comet ML experiments.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import comet_ml

from src.visualization.common import extract_algorithm_name, normalize_params, to_float


class Fetcher:
    """Fetch experiments from Comet ML and save raw metrics."""

    FETCHED_FILE = ".fetched_experiments.json"

    def __init__(self, api_key: str, workspace: Optional[str] = None):
        if not api_key:
            raise ValueError("API key required")
        self.api_key = api_key
        self.workspace = workspace
        self.api = comet_ml.API(api_key=api_key)

    def list_experiments(self, project: str):
        """List all experiments for a project."""
        return self.api.get_experiments(workspace=self.workspace, project_name=project)

    def _load_fetched_ids(self, project_dir: Path) -> Set[str]:
        """Load set of already-fetched experiment IDs."""
        fetched_file = project_dir / self.FETCHED_FILE
        if not fetched_file.exists():
            return set()
        try:
            with open(fetched_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return set(data.get("experiment_ids", []))
        except Exception:
            return set()

    def _save_fetched_ids(self, project_dir: Path, fetched_ids: Set[str]) -> None:
        """Save set of fetched experiment IDs."""
        fetched_file = project_dir / self.FETCHED_FILE
        with open(fetched_file, "w", encoding="utf-8") as fh:
            json.dump({"experiment_ids": sorted(list(fetched_ids))}, fh, indent=2)

    def fetch_project(self, project: str, metric_keys: List[str], out_dir: Path = Path("data/raw"), force: bool = False, hyperparameter_keys: Optional[List[str]] = None):
        """
        Fetch experiments for a project and save raw metrics.
        Only fetches new experiments not previously fetched.

        Args:
            project: Comet ML project name
            metric_keys: List of metric names to extract
            out_dir: Base directory to save raw data
            force: If True, re-fetch all experiments
            hyperparameter_keys: List of hyperparameter names to extract (e.g., ["batch-range"])
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

        # Load previously fetched IDs
        fetched_ids = set() if force else self._load_fetched_ids(project_dir)
        total_exps = len(experiments)
        new_count = 0

        for exp in experiments:
            try:
                exp_id = exp.key if hasattr(exp, 'key') else str(exp)
                
                # Skip if already fetched
                if exp_id in fetched_ids and not force:
                    continue

                new_count += 1

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

                # Find next available filename
                counter = 1
                while True:
                    filename = project_dir / f"experiment_{counter}.json"
                    if not filename.exists():
                        break
                    counter += 1

                # Save raw data (including hyperparameters if available)
                raw_data = {"parameters": params, "metrics": filtered}
                
                # Try to extract hyperparameters (batch-range logged as hyperparameter)
                try:
                    hyperparams = exp.get_parameters_summary() if hasattr(exp, 'get_parameters_summary') else {}
                except Exception:
                    hyperparams = {}
                
                if hyperparams:
                    raw_data["hyperparameters"] = hyperparams
                
                with open(filename, "w", encoding="utf-8") as fh:
                    json.dump(raw_data, fh, indent=2)

                fetched_ids.add(exp_id) # pyright: ignore[reportArgumentType]
                print(f"  Fetched {filename} (metrics={len(filtered)})")
            except Exception as e:
                print(f"  Failed processing experiment {exp}: {e}")

        # Save updated fetched IDs
        self._save_fetched_ids(project_dir, fetched_ids)
        print(f"  Total: {total_exps} experiments, {new_count} new, {len(fetched_ids)} cached")


class Merger:
    """Merge raw experimental data into aggregated metrics by algorithm."""

    def __init__(self, raw_dir: Path = Path("data/raw"), merge_dir: Path = Path("data/merge"), batch_range_categories: Optional[Dict[str, float]] = None):
        self.raw_dir = Path(raw_dir)
        self.merge_dir = Path(merge_dir)
        self.batch_range_categories = batch_range_categories or {}

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

        # Initialize merged structure: {batch_range: {metric_name: {algorithm: [run1_points, run2_points, ...]}}}
        merged_by_range: Dict[str, Dict[str, Dict[str, List[List[Dict[str, Any]]]]]] = {}

        # Process each raw experiment file
        for i, raw_file in enumerate(sorted(project_raw.glob("*.json")), start=1):
            try:
                with open(raw_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                params = data.get("parameters", {})
                hyperparams = data.get("hyperparameters", {})
                metrics = data.get("metrics", [])

                # Extract algorithm name
                norm_params = normalize_params(params)
                alg_name = extract_algorithm_name(norm_params, i)

                # Determine batch range category from hyperparameter
                norm_hyperparams = normalize_params(hyperparams)
                batch_range_value = norm_hyperparams.get("batch-range") or norm_hyperparams.get("batch_range")
                batch_range_value = to_float(batch_range_value)
                batch_range = self._categorize_batch_size(batch_range_value)

                # Initialize batch range if needed
                if batch_range not in merged_by_range:
                    merged_by_range[batch_range] = {k: {} for k in metric_keys}

                # Process each metric
                for metric_key in metric_keys:
                    metric_points = [m for m in metrics if m.get("metricName") == metric_key]
                    run_points = self._extract_points(metric_points)

                    if run_points:  # Only store if we have data
                        merged_by_range[batch_range][metric_key].setdefault(alg_name, []).append(run_points)

            except Exception as e:
                print(f"  Failed processing raw file {raw_file}: {e}")

        # Write merged metrics organized by batch range
        success = True
        for batch_range, range_data in merged_by_range.items():
            # Create batch range subdirectory
            batch_range_dir = project_merge / batch_range
            batch_range_dir.mkdir(parents=True, exist_ok=True)

            for metric_key in metric_keys:
                try:
                    merged_metric = range_data.get(metric_key, {})
                    out_file = batch_range_dir / f"{metric_key}.json"
                    with open(out_file, "w", encoding="utf-8") as fh:
                        json.dump(merged_metric, fh, indent=2)
                    print(f"  Wrote merged metric {metric_key} to {out_file}")
                except Exception as e:
                    print(f"  Failed to write merged metric {metric_key} for {project}/{batch_range}: {e}")
                    success = False

        return success

    def _categorize_batch_size(self, batch_size: Optional[float]) -> str:
        """
        Categorize batch size into a category based on batch_range_categories.
        Matches batch_size <= threshold to find the appropriate category.
        Defaults to 'unknown' if no match or no batch_size provided.
        """
        if batch_size is None or not self.batch_range_categories:
            return "unknown"

        # Convert thresholds to floats (handles both numeric and string values from YAML)
        categories_as_floats = []
        for category, threshold in self.batch_range_categories.items():
            threshold_float = to_float(threshold)
            if threshold_float is not None:
                categories_as_floats.append((category, threshold_float))

        # Sort categories by threshold ascending to check from smallest to largest
        sorted_categories = sorted(categories_as_floats, key=lambda x: x[1])

        for category, threshold in sorted_categories:
            if batch_size <= threshold:
                return category

        # If batch_size is larger than all thresholds, return the largest category
        if sorted_categories:
            return sorted_categories[-1][0]

        return "unknown"

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
