#!/usr/bin/env python3
"""
Fetch metrics from Comet ML and merge into aggregated files.

Usage:
    python fetch_and_merge.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.visualization.config import ConfigManager
from src.visualization.data import Fetcher, Merger  # noqa: F401


def main():
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

    if not metrics or not projects:
        print(
            "Error: metric_keys and projects required in config/visualization.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch and merge
    print(f"Workspace: {workspace}")
    print(f"Metrics: {len(metrics)} metrics")
    print(f"Projects: {len(projects)} projects\n")

    fetcher = Fetcher(api_key=api_key, workspace=workspace)
    merger = Merger(
        raw_dir=cfg.directories()["raw_dir"], merge_dir=cfg.directories()["merge_dir"]
    )

    for project in projects:
        print(f"Processing: {project}")
        try:
            fetcher.fetch_project(
                project=project,
                metric_keys=metrics,
                out_dir=cfg.directories()["raw_dir"],
            )
            merger.merge_project(project, metrics)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
