#!/usr/bin/env python3
"""
Fetch metrics from Comet ML and merge into aggregated files.

Usage:
    python fetch_and_merge.py --benchmark-type all
    python fetch_and_merge.py --benchmark-type dynamic --force
    python fetch_and_merge.py --benchmark-type static --skip-merge
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.visualization.config import ConfigManager
from src.visualization.data import Fetcher, Merger  # noqa: F401


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fetch metrics from Comet ML and merge")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all experiments (by default only fetches if raw data doesn't exist)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only fetch data, skip merging step",
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["dynamic", "static", "all"],
        default="all",
        help="Which benchmark mode to process",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv(dotenv_path=".env")

    # Get configuration
    cfg = ConfigManager()

    # Get credentials
    api_key = os.getenv("COMET_API_KEY")
    workspace = cfg.workspace() or os.getenv("COMET_WORKSPACE")

    if not api_key:
        print("Error: COMET_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if not workspace:
        print("Error: COMET_WORKSPACE not set", file=sys.stderr)
        sys.exit(1)

    print(f"Workspace: {workspace}")
    print(f"Mode: {'Force re-fetch all' if args.force else 'Fetch only new experiments'}")
    if args.skip_merge:
        print("Merge: Skipped")
    print()

    fetcher = Fetcher(api_key=api_key, workspace=workspace)

    benchmark_types = cfg.available_modes() if args.benchmark_type == "all" else [args.benchmark_type]

    for benchmark_type in benchmark_types:
        metrics = cfg.metric_keys(benchmark_type)
        projects = cfg.projects(benchmark_type)
        hyperparameters = cfg.hyperparameters(benchmark_type)

        if not metrics or not projects:
            print(f"Skipping {benchmark_type}: missing metric_keys or projects in config")
            continue

        directories = cfg.directories(benchmark_type)
        raw_dir = Path(directories["raw_dir"])
        merge_dir = Path(directories["merge_dir"])
        batch_range_categories = cfg.batch_range_categories(benchmark_type)
        use_batch_ranges = cfg.uses_batch_ranges(benchmark_type)
        merger = Merger(raw_dir=raw_dir, merge_dir=merge_dir, batch_range_categories=batch_range_categories)

        print(f"Benchmark type: {benchmark_type}")
        print(f"Metrics: {len(metrics)} metrics")
        print(f"Projects: {len(projects)} projects")
        print(f"Hyperparameters: {len(hyperparameters)} hyperparameters")

        for project in projects:
            print(f"Processing: {project}")

            try:
                fetcher.fetch_project(
                    project=project,
                    metric_keys=metrics,
                    out_dir=raw_dir,
                    force=args.force,
                    hyperparameter_keys=hyperparameters,
                )
            except Exception as e:
                print(f"  Fetch error: {e}")
                continue

            if not args.skip_merge:
                try:
                    merger.merge_project(project, metrics, use_batch_ranges=use_batch_ranges)
                except Exception as e:
                    print(f"  Merge error: {e}")

        print()


if __name__ == "__main__":
    main()
