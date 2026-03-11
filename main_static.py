"""
Static graph benchmarking entry point.

Treats a static graph as a TemporalGraph with steps=[] (1 snapshot).
Reuses the existing temporal pipeline: run_algorithm -> evaluate -> log_results.

Usage:
    python main_static.py --dataset karate --dataset-path data/karate.txt
    python main_static.py --dataset email-enron-large --dataset-path data/email-enron-large.txt
    python main_static.py --builtin karate
    python main_static.py --list-builtins
    python main_static.py --config email-enron-large
"""
import argparse
import os
import sys

import comet_ml  # noqa: F401 — must be imported before Experiment
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.algorithms.factory import load_algorithms
from src.dataloader.static_loader import load_static_as_temporal, load_builtin_graph, BUILTIN_GRAPHS
from src.pipeline_utils import run_algorithm, evaluate, log_results

load_dotenv()

CONFIG_PATH = "config/static_dataset_config.yaml"


def parse_static_args():
    parser = argparse.ArgumentParser(
        description="Static Graph Community Detection Benchmarking"
    )

    # Data source (mutually exclusive: file path, built-in, or config)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to edge list / CSV / GML file",
    )
    source.add_argument(
        "--builtin", type=str, default=None,
        help="Load a built-in NetworkX graph (e.g., 'karate')",
    )
    source.add_argument(
        "--config", type=str, default=None,
        help="Load dataset by name from config/static_dataset_config.yaml datasets section",
    )

    # Dataset metadata
    parser.add_argument(
        "--dataset", type=str, default="static",
        help="Dataset name for logging (default: 'static')",
    )

    # Edge list parsing
    parser.add_argument("--source-idx", type=int, default=0, help="Source column index (default: 0)")
    parser.add_argument("--target-idx", type=int, default=1, help="Target column index (default: 1)")
    parser.add_argument(
        "--delimiter", type=str, default=" ",
        help="Field delimiter (default: space). Use 'tab' for TSV files.",
    )
    parser.add_argument(
        "--preload-fraction", type=float, default=None,
        help="Fraction of edges to load (default: config value or 1.0)",
    )
    parser.add_argument(
        "--ground-truth-attr", type=str, default=None,
        help="Node attribute for ground truth communities",
    )

    # Run settings
    parser.add_argument(
        "-n", "--num-runs", type=int, default=5,
        help="Number of runs per algorithm (default: 5)",
    )

    # Info flags
    parser.add_argument("--list-builtins", action="store_true", help="List available built-in graphs")
    parser.add_argument("--list-datasets", action="store_true", help="List static datasets from config")

    return parser.parse_args()


def _load_static_config(dataset_key: str) -> dict:
    """Load a static graph config entry from static_dataset_config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    if dataset_key not in datasets:
        available = ", ".join(sorted(datasets.keys())) if datasets else "(none defined)"
        print(f"Error: Static dataset '{dataset_key}' not found. Available: {available}")
        sys.exit(1)

    return datasets[dataset_key]


def main() -> None:
    args = parse_static_args()

    # Info flags
    if args.list_builtins:
        print("Available built-in graphs:")
        for name in sorted(BUILTIN_GRAPHS.keys()):
            print(f"  {name}")
        return

    if args.list_datasets:
        if not os.path.exists(CONFIG_PATH):
            print(f"Config not found: {CONFIG_PATH}")
            return
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        datasets = config.get("datasets", {})
        target = config.get("target_datasets", list(datasets.keys()))
        print("Static datasets (from config):")
        for name in target:
            entry = datasets.get(name, {})
            path = entry.get("path", "?")
            print(f"  {name}: {path}")
        return

    # Load graph
    if args.builtin:
        tg = load_builtin_graph(args.builtin)
        dataset_name = args.dataset if args.dataset != "static" else args.builtin
    elif args.config:
        cfg = _load_static_config(args.config)
        delimiter = cfg.get("delimiter", " ")
        if delimiter == "tab":
            delimiter = "\t"
        preload_fraction = args.preload_fraction
        if preload_fraction is None:
            preload_fraction = cfg.get("preload_fraction", 1.0)
        tg = load_static_as_temporal(
            file_path=cfg["path"],
            source_idx=cfg.get("source_idx", 0),
            target_idx=cfg.get("target_idx", 1),
            delimiter=delimiter,
            preload_fraction=preload_fraction,
            ground_truth_attr=cfg.get("ground_truth_attr"),
        )
        dataset_name = args.dataset if args.dataset != "static" else cfg.get("dataset_name", args.config)
    elif args.dataset_path:
        delimiter = args.delimiter
        if delimiter == "tab":
            delimiter = "\t"
        preload_fraction = args.preload_fraction if args.preload_fraction is not None else 1.0
        tg = load_static_as_temporal(
            file_path=args.dataset_path,
            source_idx=args.source_idx,
            target_idx=args.target_idx,
            delimiter=delimiter,
            preload_fraction=preload_fraction,
            ground_truth_attr=args.ground_truth_attr,
        )
        dataset_name = args.dataset
    else:
        print("Error: Provide one of --dataset-path, --builtin, or --config")
        sys.exit(1)

    # Summary
    G = tg.base_graph
    print(f"Static graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Dataset: {dataset_name}")
    has_gt = tg._ground_truth_clusterings is not None
    print(f"Ground truth: {'yes' if has_gt else 'no'}")
    effective_preload_fraction = args.preload_fraction
    if args.config and effective_preload_fraction is None:
        effective_preload_fraction = cfg.get("preload_fraction", 1.0)
    if args.dataset_path and effective_preload_fraction is None:
        effective_preload_fraction = 1.0
    if effective_preload_fraction is not None and effective_preload_fraction < 1.0:
        print(f"Preload fraction: {effective_preload_fraction}")
    print()

    # Load algorithms from config — filter to static-only (dynamic algorithms
    # require temporal steps and don't make sense for a single snapshot)
    all_algorithms = load_algorithms("config/algorithms.yaml")
    algorithms = {
        name: entry for name, entry in all_algorithms.items()
        if entry["type"] == "static"
    }
    skipped = set(all_algorithms.keys()) - set(algorithms.keys())
    if skipped:
        print(f"Skipping dynamic algorithms (not applicable to static graphs): {', '.join(sorted(skipped))}")
        print()

    # Create a namespace-like object for log_results compatibility
    # log_results expects args with temporal-specific fields; provide sensible defaults
    class StaticArgs:
        pass

    static_args = StaticArgs()
    static_args.dataset = dataset_name
    static_args.benchmark_mode = "static"
    static_args.max_steps = 0
    static_args.batch_range = 0.0
    static_args.initial_fraction = 1.0
    static_args.delete_insert_ratio = 0.0
    static_args.preload_fraction = effective_preload_fraction if effective_preload_fraction is not None else 1.0

    # Run pipeline
    for _run_idx in tqdm(range(args.num_runs), desc="Runs"):
        for algo_name, algo_entry in tqdm(algorithms.items(), desc="Algorithms", leave=False):
            wrapper = algo_entry["wrapper"]
            algo_type = algo_entry["type"]
            clustering_type = algo_entry["clustering_type"]
            algo_params = algo_entry["params"]

            # Step 1: Run algorithm -> List[NodeClustering] + runtimes
            clusterings, runtimes = run_algorithm(
                wrapper, tg, algo_params, algo_type, clustering_type
            )

            # Step 2: Evaluate -> MethodDynamicResults
            results = evaluate(tg, clusterings, clustering_type, runtimes)

            # Step 3: Log results
            log_results(
                results, algo_name, algo_type, clustering_type,
                algo_params, tg, static_args,
            )

            # Print summary
            print(f"{algo_name}:")
            print(f"  nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
            print(f"  runtime: {results.avg_runtime:.4f}s")
            print(f"  cdlib_modularity_overlap: {results.avg_cdlib_modularity_overlap:.4f}")
            print(f"  customize_q0_overlap: {results.avg_customize_q0_overlap:.4f}")
            if results.nmi_trace:
                print(f"  nmi: {results.avg_nmi:.4f}")
            print()


if __name__ == "__main__":
    main()
