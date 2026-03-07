"""
Pipeline utilities for running community detection benchmarks.

Three simple functions:
- run_algorithm(): executes algorithm, returns List[NodeClustering]
- evaluate(): computes metrics per snapshot
- log_results(): logs to Comet ML
"""
import os
import time
from typing import Any, List

from cdlib import NodeClustering
from comet_ml import Experiment
from dotenv import load_dotenv

from src.evaluations.metrics import compute_modularity, compute_nmi_from_ground_truth
from src.factory.communities import IntermediateResults, MethodDynamicResults
from src.factory.factory import TemporalGraph

load_dotenv()


def run_algorithm(
    wrapper: Any,
    tg: TemporalGraph,
    params: dict,
    algo_type: str,
    clustering_type: str,
) -> tuple[List[NodeClustering], list[float]]:
    """
    Execute algorithm on temporal graph.
    
    Normalizes wrapper output to always return List[NodeClustering].
    Also returns runtime per snapshot.
    
    Args:
        wrapper: Algorithm wrapper (StaticMethodWrapper, DynamicMethodWrapper, 
                 or CommunityDetectionAlgorithm class/instance)
        tg: TemporalGraph with snapshots
        params: Algorithm parameters
        algo_type: "static" or "dynamic"
        clustering_type: "crisp" or "overlapping"
    
    Returns:
        Tuple of (List[NodeClustering], runtime_list)
    """
    runtimes = []
    
    # Use the wrapper's __call__ method to get results
    # The wrapper handles iteration for static methods
    start_total = time.perf_counter()
    result = wrapper(tg, **params)
    elapsed_total = time.perf_counter() - start_total
    
    if algo_type == "static":
        # Static: wrapper returns List[NodeClustering], distribute runtime evenly
        if isinstance(result, list):
            clusterings = result
            # Distribute total runtime across snapshots
            runtimes = [elapsed_total / len(clusterings)] * len(clusterings)
        else:
            clusterings = [result]
            runtimes = [elapsed_total]
    else:
        # Dynamic: may return MethodDynamicResults or List[NodeClustering]
        if isinstance(result, MethodDynamicResults):
            clusterings = result.clusterings
            runtimes = result.runtimes if result.runtimes else [0.0] * len(tg)
        elif isinstance(result, list):
            clusterings = result
            runtimes = [elapsed_total / len(clusterings)] * len(clusterings)
        else:
            clusterings = [result]
            runtimes = [elapsed_total]
    
    return clusterings, runtimes


def evaluate(
    tg: TemporalGraph,
    clusterings: List[NodeClustering],
    clustering_type: str,
    runtimes: list[float] | None = None,
    ground_truth_attr: str | None = None,
) -> MethodDynamicResults:
    """
    Compute metrics for each snapshot.
    
    Args:
        tg: TemporalGraph with snapshots
        clusterings: List of NodeClustering, one per snapshot
        clustering_type: "crisp" or "overlapping"
        runtimes: Optional runtime per snapshot (if already measured)
        ground_truth_attr: Node attribute for ground truth (if available)
    
    Returns:
        MethodDynamicResults with all metric traces
    """
    results = MethodDynamicResults()
    
    # Check if ground truth is available
    has_ground_truth = ground_truth_attr is not None and hasattr(tg, '_ground_truth_attr')
    
    for i, snapshot in enumerate(tg.iter_snapshots()):
        if i >= len(clusterings):
            break
            
        communities = clusterings[i]
        
        # Use provided runtime or default
        runtime = runtimes[i] if runtimes and i < len(runtimes) else 0.0
        
        # Compute modularity metrics
        cdlib_mod, q0_mod = compute_modularity(snapshot, communities, clustering_type)
        
        results.update_intermediate_results(
            IntermediateResults(
                runtime=runtime,
                cdlib_modularity_overlap=cdlib_mod,
                customize_q0_overlap=q0_mod,
                affected_nodes=snapshot.number_of_nodes(),
                num_communities=len(communities.communities),
            )
        )
        results.clusterings.append(communities)
        
        # Compute NMI if ground truth is available
        if has_ground_truth:
            nmi = compute_nmi_from_ground_truth(
                snapshot, communities, ground_truth_attr
            )
            # Add NMI to results if the attribute exists
            if not hasattr(results, 'nmi_trace'):
                results.nmi_trace = []
            results.nmi_trace.append(nmi)
    
    return results


def log_results(
    results: MethodDynamicResults,
    algo_name: str,
    algo_type: str,
    clustering_type: str,
    algo_params: dict,
    tg: TemporalGraph,
    args,
) -> None:
    """
    Log results to Comet ML.
    
    Args:
        results: MethodDynamicResults with metrics
        algo_name: Algorithm name
        algo_type: "static" or "dynamic"
        clustering_type: "crisp" or "overlapping"
        algo_params: Algorithm hyperparameters
        tg: TemporalGraph for metadata
        args: CLI arguments
    """
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=f"graph-community-detection-overlapping-{args.dataset.lower()}",
        workspace=os.getenv("COMET_WORKSPACE"),
    )

    experiment.add_tag(algo_name)
    experiment.add_tag(args.dataset)

    # Base parameters
    params_to_log = {
        "algorithm": algo_name,
        "dataset": args.dataset,
        "algorithm_type": algo_type,
        "clustering_type": clustering_type,
        "num_snapshots": len(tg),
        "initial_nodes": tg.base_graph.number_of_nodes(),
        "initial_edges": tg.base_graph.number_of_edges(),
        "max_steps": args.max_steps,
        "batch_range": args.batch_range,
        "initial_fraction": args.initial_fraction,
        "delete_insert_ratio": args.delete_insert_ratio,
    }
    # Algorithm hyperparameters
    params_to_log.update(algo_params)
    experiment.log_parameters(params_to_log)

    # Summary metrics
    summary_metrics = {
        "avg_runtime": results.avg_runtime,
        "total_runtime": results.total_runtime,
        "avg_cdlib_modularity_overlap": results.avg_cdlib_modularity_overlap,
        "cdlib_modularity_overlap_stability": results.cdlib_modularity_overlap_stability,
        "customize_q0_overlap_stability": results.customize_q0_overlap_stability,
        "avg_customize_q0_overlap": results.avg_customize_q0_overlap,
        "num_steps": len(results.runtimes),
    }
    
    # Add NMI if available
    if results.nmi_trace:
        summary_metrics["avg_nmi"] = results.avg_nmi
    
    experiment.log_metrics(summary_metrics)

    # Per-step metrics
    has_nmi = bool(results.nmi_trace)
    for step, (runtime, cdlib_mod, q0_mod, num_comm) in enumerate(zip(
        results.runtimes,
        results.cdlib_modularity_overlap_trace,
        results.customize_q0_overlap_trace,
        results.num_communities,
    )):
        step_metrics = {
            "runtime": runtime,
            "cdlib_modularity": cdlib_mod,
            "customize_q0_modularity": q0_mod,
            "num_communities": num_comm,
        }
        if has_nmi:
            step_metrics["nmi"] = results.nmi_trace[step]
        
        experiment.log_metrics(step_metrics, step=step)

    experiment.end()
