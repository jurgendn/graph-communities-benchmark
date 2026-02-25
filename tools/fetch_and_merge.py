#!/usr/bin/env python3
"""
Fetch metrics from Comet ML and merge into aggregated files.

Usage:
    python fetch_and_merge.py              # Fetch only new experiments
    python fetch_and_merge.py --force      # Re-fetch all experiments
    python fetch_and_merge.py --skip-merge # Only fetch, skip merging
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

    # Get what to process
    metrics = cfg.metric_keys()
    projects = cfg.projects()
    hyperparameters = cfg.hyperparameters()

    if not metrics or not projects:
        print(
            "Error: metric_keys and projects required in config/visualization.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch and merge
    print(f"Workspace: {workspace}")
    print(f"Metrics: {len(metrics)} metrics")
    print(f"Projects: {len(projects)} projects")
    print(f"Hyperparameters: {len(hyperparameters)} hyperparameters")
    print(f"Mode: {'Force re-fetch all' if args.force else 'Fetch only new experiments'}")
    if args.skip_merge:
        print("Merge: Skipped")
    print()

    raw_dir = Path(cfg.directories()["raw_dir"])
    merge_dir = Path(cfg.directories()["merge_dir"])
    batch_range_categories = cfg.batch_range_categories()

    fetcher = Fetcher(api_key=api_key, workspace=workspace)
    merger = Merger(raw_dir=raw_dir, merge_dir=merge_dir, batch_range_categories=batch_range_categories)

    for project in projects:
        print(f"Processing: {project}")

        # Fetch (only new experiments unless --force is set)
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

        # Merge
        if not args.skip_merge:
            try:
                merger.merge_project(project, metrics)
            except Exception as e:
                print(f"  Merge error: {e}")


if __name__ == "__main__":
    main()

