"""Main entry point for running the benchmark with Comet ML logging."""

import os
from pathlib import Path

from comet_ml import Experiment
from dotenv import load_dotenv

from src.benchmark import run_dynamic_benchmark
from src.dataloader.data_reader import load_txt_dataset

# Load environment variables
load_dotenv()


def main():
    dataset_name = "CollegeMsg"
    dataset_path = f"data/{dataset_name}.txt"

    tg = load_txt_dataset(
        file_path=dataset_path,
        source_idx=0,
        target_idx=1,
        batch_range=1e-4,
        initial_fraction=0.4,
        max_steps=10,
        load_full_nodes=False,
    )

    print(f"Loaded temporal graph: {len(tg)} snapshots")
    print(f"Base graph: {tg.base_graph.number_of_nodes()} nodes, {tg.base_graph.number_of_edges()} edges\n")

    results = run_dynamic_benchmark(tg)

    # Log each algorithm's results to Comet ML
    for algo_name, res in results.items():
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME", "graph-communities-benchmark"),
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
        })

        # Log summary metrics
        experiment.log_metrics({
            "avg_runtime": res.avg_runtime,
            "total_runtime": res.total_runtime,
            "avg_modularity": res.avg_modularities,
            "modularity_stability": res.modularity_stability,
            "num_steps": len(res.runtimes),
        })

        # Log per-step metrics
        for step, (runtime, modularity, num_communities) in enumerate(
            zip(res.runtimes, res.modularities, res.num_communities)
        ):
            experiment.log_metrics({
                "runtime": runtime,
                "modularity": modularity,
                "num_communities": num_communities,
            }, step=step)

        experiment.end()

        print(f"{algo_name}:")
        print(f"  steps: {len(res.runtimes)}")
        print(f"  avg_runtime: {res.avg_runtime:.4f}s")
        print(f"  avg_modularity: {res.avg_modularities:.4f}")
        print()


if __name__ == "__main__":
    main()
