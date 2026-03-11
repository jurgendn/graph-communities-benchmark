"""
Unified entry point for the Graph Communities Benchmark.

Supports both static algorithms (per-snapshot) and dynamic algorithms
(full temporal graph). Uses a clean 3-step pipeline:
1. Run algorithm → List[NodeClustering]
2. Evaluate → MethodDynamicResults  
3. Log results → Comet ML

Usage:
    python main.py [--dataset CollegeMsg] [--max-steps 10] ...
"""
import os

import comet_ml  # noqa: F401 — must be imported before Experiment
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.algorithms.factory import load_algorithms
from src.dataloader.data_reader import load_txt_dataset, load_lfr_folder
from src.pipeline_utils import run_algorithm, evaluate, log_results
from src.utils.arg_parser import parse_args

load_dotenv()


def main() -> None:
    args = parse_args()

    # Step 0: Load data
    if args.lfr_folder:
        # Load LFR benchmark from folder
        tg = load_lfr_folder(
            folder_path=args.lfr_folder,
            ground_truth_attr=args.ground_truth_attr,
            max_steps=args.max_steps,
        )
        print(f"Loaded LFR benchmark: {len(tg)} snapshots")
    else:
        # Load from edge-list file
        tg = load_txt_dataset(
            file_path=args.dataset_path,
            source_idx=args.source_idx,
            target_idx=args.target_idx,
            batch_range=args.batch_range,
            initial_fraction=args.initial_fraction,
            max_steps=args.max_steps,
            load_full_nodes=args.load_full_nodes,
            delimiter=args.delimiter,
            delete_insert_ratio=args.delete_insert_ratio,
        )
        print(f"Loaded temporal graph: {len(tg)} snapshots")

    print(
        f"Base graph: {tg.base_graph.number_of_nodes()} nodes, "
        f"{tg.base_graph.number_of_edges()} edges"
    )

    print(f"Average changes per snapshot: {tg.average_changes_per_snapshot():.2f}\n")

    # Load algorithms from config
    algorithms = load_algorithms("config/algorithms.yaml")

    # Run pipeline for each algorithm
    for _run_idx in tqdm(range(args.num_runs), desc="Runs"):
        for algo_name, algo_entry in tqdm(algorithms.items(), desc="Algorithms", leave=False):
            wrapper = algo_entry["wrapper"]
            algo_type = algo_entry["type"]
            clustering_type = algo_entry["clustering_type"]
            algo_params = algo_entry["params"]

            # Step 1: Run algorithm → List[NodeClustering] + runtimes
            clusterings, runtimes = run_algorithm(
                wrapper, tg, algo_params, algo_type, clustering_type
            )

            # Step 2: Evaluate → MethodDynamicResults
            results = evaluate(
                tg, clusterings, clustering_type, runtimes
            )

            # Step 3: Log results
            log_results(
                results,
                algo_name,
                algo_type,
                clustering_type,
                algo_params,
                tg,
                args,
            )

            # Print summary
            print(f"{algo_name}:")
            print(f"  steps: {len(results.runtimes)}")
            print(f"  avg_runtime: {results.avg_runtime:.4f}s")
            print(f"  avg_cdlib_modularity_overlap: {results.avg_cdlib_modularity_overlap:.4f}")
            print(f"  avg_customize_q0_overlap: {results.avg_customize_q0_overlap:.4f}")
            if results.nmi_trace:
                print(f"  avg_nmi: {results.avg_nmi:.4f}")
            print()


if __name__ == "__main__":
    main()
