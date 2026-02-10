"""Main entry point for running the benchmark with Comet ML logging."""

import os

from comet_ml import Experiment
from dotenv import load_dotenv

from src.benchmark import run_dynamic_benchmark
from src.dataloader.data_reader import load_txt_dataset
from src.utils.arg_parser import parse_args

# Load environment variables
load_dotenv()


def main():
    args = parse_args()
    dataset_name = args.dataset

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
    print(f"Base graph: {tg.base_graph.number_of_nodes()} nodes, {tg.base_graph.number_of_edges()} edges\n")

    results = run_dynamic_benchmark(tg)

    # Log each algorithm's results to Comet ML
    for algo_name, res in results.items():
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=f"graph-community-detection-overlapping-{args.dataset.lower()}",
            workspace=os.getenv("COMET_WORKSPACE"),
        )

        # Tag experiment with method and dataset for aggregation
        experiment.add_tag(algo_name)
        experiment.add_tag(dataset_name)

        # Log metadata
        experiment.log_parameters({
            "algorithm": algo_name,
            "dataset": dataset_name,
            "num_snapshots": len(tg),
            "initial_nodes": tg.base_graph.number_of_nodes(),
            "initial_edges": tg.base_graph.number_of_edges(),
            "max_steps": args.max_steps,
            "batch_range": args.batch_range,
            "initial_fraction": args.initial_fraction,
            "delete_insert_ratio": args.delete_insert_ratio,
        })

        # Log summary metrics
        experiment.log_metrics(
            {
                "avg_runtime": res.avg_runtime,
                "total_runtime": res.total_runtime,
                "avg_cdlib_modularity_overlap": res.avg_cdlib_modularity_overlap,
                "cdlib_modularity_overlap_stability": res.cdlib_modularity_overlap_stability,
                "customize_q0_overlap_stability": res.customize_q0_overlap_stability,
                "avg_customize_q0_overlap": res.avg_customize_q0_overlap,
                "num_steps": len(res.runtimes),
            }
        )

        # Log per-step metrics
        for step, (runtime, cdlib_modularity, customize_q0_modularity, num_communities) in enumerate(
            zip(res.runtimes, res.cdlib_modularity_overlap_trace, res.customize_q0_overlap_trace, res.num_communities)
        ):
            experiment.log_metrics(
                {
                    "runtime": runtime,
                    "cdlib_modularity": cdlib_modularity,
                    "customize_q0_modularity": customize_q0_modularity,
                    "num_communities": num_communities,
                },
                step=step,
            )

        experiment.end()

        print(f"{algo_name}:")
        print(f"  steps: {len(res.runtimes)}")
        print(f"  avg_runtime: {res.avg_runtime:.4f}s")
        print(f"  avg_cdlib_modularity_overlap: {res.avg_cdlib_modularity_overlap:.4f}")
        print(f"  avg_customize_q0_overlap: {res.avg_customize_q0_overlap:.4f}")
        print()


if __name__ == "__main__":
    main()
