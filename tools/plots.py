#!/usr/bin/env python3
"""
Generate grouped multi-dataset plots.

Usage:
    python plot_group.py [--metric NAME] [--out DIR]

Examples:
    python plot_group.py                    # All metrics
    python plot_group.py --metric modularity  # Specific metric
"""
import argparse
import sys
from pathlib import Path

from src.visualization.config import ConfigManager
from src.visualization.core import GroupedPlotter


def main():
    parser = argparse.ArgumentParser(description="Generate grouped plots")
    parser.add_argument("--metric", "-m", help="Metric name (optional)")
    parser.add_argument("--merge-root", "-r", default="experiments/merged", help="Data directory")
    parser.add_argument("--out", "-o", default="assets/grouped", help="Output directory")
    args = parser.parse_args()

    # Verify merge root exists
    merge_root = Path(args.merge_root)
    if not merge_root.exists():
        print(f"Error: {merge_root} not found", file=sys.stderr)
        sys.exit(1)

    # Get config and determine metrics
    cfg = ConfigManager()
    if args.metric:
        metrics = [args.metric]
    else:
        metrics = cfg.metric_keys() or ["num_communities", "cdlib_modularity", "customize_q0_modularity", "runtime"]

    # Generate plots
    plotter = GroupedPlotter(
        merge_root=cfg.directories()["merge_dir"],
        out_base=cfg.directories()["output_dir"],
    )
    for metric in metrics:
        try:
            plotter.plot(metric)
            print(f"✓ {metric}")
        except Exception as e:
            print(f"✗ {metric}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
