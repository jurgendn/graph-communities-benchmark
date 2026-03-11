#!/usr/bin/env python3
"""
Generate grouped multi-dataset plots.

Usage:
    python plots.py --benchmark-type all [--metric NAME]

Examples:
    python plots.py --benchmark-type all
    python plots.py --benchmark-type static --metric runtime
"""
import argparse
import sys
from pathlib import Path

from src.visualization.config import ConfigManager
from src.visualization.core import GroupedPlotter


def main():
    parser = argparse.ArgumentParser(description="Generate grouped plots")
    parser.add_argument("--metric", "-m", help="Metric name (optional)")
    parser.add_argument(
        "--benchmark-type",
        choices=["dynamic", "static", "all"],
        default="all",
        help="Which benchmark mode to plot",
    )
    args = parser.parse_args()

    cfg = ConfigManager()

    benchmark_types = cfg.available_modes() if args.benchmark_type == "all" else [args.benchmark_type]
    had_input = False

    for benchmark_type in benchmark_types:
        directories = cfg.directories(benchmark_type)
        merge_root = Path(directories["merge_dir"])
        if not merge_root.exists():
            print(f"Skipping {benchmark_type}: {merge_root} not found", file=sys.stderr)
            continue

        had_input = True
        if args.metric:
            metrics = [args.metric]
        else:
            metrics = cfg.metric_keys(benchmark_type) or ["num_communities", "cdlib_modularity", "customize_q0_modularity", "runtime"]

        plotter = GroupedPlotter(
            benchmark_type=benchmark_type,
            merge_root=merge_root,
            out_base=directories["output_dir"],
            use_batch_ranges=cfg.uses_batch_ranges(benchmark_type),
        )
        for metric in metrics:
            try:
                plotter.plot(metric)
                print(f"✓ {benchmark_type}: {metric}")
            except Exception as e:
                print(f"✗ {benchmark_type}: {metric}: {e}", file=sys.stderr)

    if not had_input:
        print("Error: no merged experiment directories found for the selected benchmark type(s)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
