import importlib
import os
import time

import comet_ml  # noqa: F401
import yaml
from cdlib import NodeClustering, evaluation
from comet_ml import Experiment
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.dataloader.data_reader import load_txt_dataset
from src.evaluations.target_modularity import overlapping_modularity_q0
from src.factory.communities import IntermediateResults, MethodDynamicResults
from src.utils.arg_parser import parse_args

# Load environment variables
load_dotenv()


def load_algorithms_config(config_path: str = "config/algorithms.yaml"):
    """Load algorithm configuration from YAML file and resolve function references using importlib."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    target_algorithms = config.get("target_algorithms", [])
    all_algorithms = config.get("overlapping_algorithms", {})
    
    # Build the algorithms dictionary with resolved function references
    algorithms_dict = {}
    for algo_name in target_algorithms:
        if algo_name not in all_algorithms:
            print(f"Warning: Algorithm '{algo_name}' not found in configuration, skipping.")
            continue
        
        algo_config = all_algorithms[algo_name]
        module_path = algo_config.get("module", "cdlib.algorithms")
        func_name = algo_config["function"]
        
        # Dynamically import the module and get the function
        try:
            module = importlib.import_module(module_path)
            algo_func = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load '{func_name}' from '{module_path}': {e}, skipping '{algo_name}'.")
            continue
        
        algorithms_dict[algo_name] = {
            "func": algo_func,
            "params": algo_config.get("params", {}),
            "metadata": algo_config.get("metadata", {}),
        }
    
    return algorithms_dict


ALGORITHMS_OVERLAPPING = load_algorithms_config()

def main():
    args = parse_args()
    num_runs = args.num_runs
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
    print(f"Average changes per snapshot: {tg.average_changes_per_snapshot():.2f}\n")

    algorithm_names = ALGORITHMS_OVERLAPPING.keys()

    runs_bar = tqdm(range(num_runs), total=num_runs, desc="Runs")
    for run_idx in runs_bar:
        algorithms_bar = tqdm(
            algorithm_names, total=len(algorithm_names), desc="Algorithms", leave=False
        )
        for algo_name in algorithms_bar:
            algo_func = ALGORITHMS_OVERLAPPING[algo_name]["func"]
            algo_params = ALGORITHMS_OVERLAPPING[algo_name]["params"]
            res = MethodDynamicResults()
            steps_bar = tqdm(tg.iter_snapshots(), total=len(tg), desc="Steps")
            for step_idx, snapshot in enumerate(steps_bar):
                start_time = time.perf_counter()
                communities: NodeClustering = algo_func(snapshot, **algo_params)
                elapsed = time.perf_counter() - start_time

                q0_modularity = overlapping_modularity_q0(snapshot, communities)
                cdlib_modularity = evaluation.modularity_overlap(
                    snapshot, communities
                ).score

                intermediate = IntermediateResults(
                    runtime=elapsed,
                    cdlib_modularity_overlap=cdlib_modularity,
                    customize_q0_overlap=q0_modularity,
                    affected_nodes=snapshot.number_of_nodes(),
                    num_communities=len(communities.communities),
                )
                res.update_intermediate_results(intermediate)

            experiment = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=f"graph-community-detection-overlapping-{args.dataset.lower()}",
                workspace=os.getenv("COMET_WORKSPACE"),
            )

            # Tag experiment with method and dataset for aggregation
            experiment.add_tag(algo_name)
            experiment.add_tag(dataset_name)

            # Log metadata
            experiment.log_parameters(
                {
                    "algorithm": algo_name,
                    "dataset": dataset_name,
                    "num_snapshots": len(tg),
                    "initial_nodes": tg.base_graph.number_of_nodes(),
                    "initial_edges": tg.base_graph.number_of_edges(),
                    "max_steps": args.max_steps,
                    "batch_range": args.batch_range,
                    "initial_fraction": args.initial_fraction,
                    "delete_insert_ratio": args.delete_insert_ratio,
                }
            )

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
            for step, (
                runtime,
                cdlib_modularity,
                customize_q0_modularity,
                num_communities,
            ) in enumerate(
                zip(
                    res.runtimes,
                    res.cdlib_modularity_overlap_trace,
                    res.customize_q0_overlap_trace,
                    res.num_communities,
                )
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
            print(
                f"  avg_cdlib_modularity_overlap: {res.avg_cdlib_modularity_overlap:.4f}"
            )
            print(f"  avg_customize_q0_overlap: {res.avg_customize_q0_overlap:.4f}")
            print()


if __name__ == "__main__":
    main()
