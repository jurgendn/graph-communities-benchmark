import time

from cdlib import NodeClustering, algorithms, evaluation
from tqdm.auto import tqdm

from src import dynamic_methods
from src.evaluations.target_modularity import overlapping_modularity_q0
from src.factory.communities import IntermediateResults, MethodDynamicResults
from src.factory.factory import TemporalGraph
from src.static_methods.ndocd import NDOCD


def run_static_benchmark(
    temporal_graph: TemporalGraph,
    algorithm_names: list[str] | None = None,
) -> dict[str, MethodDynamicResults]:
    """
    Run community detection algorithms on each snapshot of a temporal graph.

    Args:
        temporal_graph: TemporalGraph with snapshots to benchmark
        algorithm_names: List of algorithm names to run. Defaults to ["louvain", "label_propagation"]

    Returns:
        Dictionary mapping algorithm name to MethodDynamicResults
    """
    ALGORITHMS_OVERLAPPING = {
        # "angel": {
        #     "func": algorithms.angel,
        #     "params": {"threshold": 0.25},
        #     "metadata": {},
        # },
        # "demon": {
        #     "func": algorithms.demon,
        #     "params": {"epsilon": 0.25},
        #     "metadata": {},
        # },
        "coach": {"func": algorithms.coach, "params": {}, "metadata": {}},
        "ndocd": {"func": NDOCD(), "params": {}, "metadata": {}},
        "percomvc": {"func": algorithms.percomvc, "params": {}, "metadata": {}},
        "slpa": {"func": algorithms.slpa, "params": {}, "metadata": {}},
        # "dcs": {"func": algorithms.dcs, "params": {}, "metadata": {}},
        # "lfm": {"func": algorithms.lfm, "params": {"alpha": 1.0}, "metadata": {}},
    }

    if algorithm_names is None:
        algorithm_names = ALGORITHMS_OVERLAPPING.keys()

    results: dict[str, MethodDynamicResults] = {
        name: MethodDynamicResults() for name in algorithm_names
    }

    algorithms_bar = tqdm(algorithm_names, total=len(algorithm_names), desc="Algorithms", leave=False)
    for algo_name in algorithms_bar:
        algo_func = ALGORITHMS_OVERLAPPING[algo_name]["func"]
        algo_params = ALGORITHMS_OVERLAPPING[algo_name]["params"]

        steps_bar = tqdm(temporal_graph.iter_snapshots(), total=len(temporal_graph), desc="Steps")  
        for step_idx, snapshot in enumerate(steps_bar):
            start_time = time.perf_counter()
            communities: NodeClustering = algo_func(snapshot, **algo_params)
            elapsed = time.perf_counter() - start_time

            q0_modularity = overlapping_modularity_q0(snapshot, communities)
            cdlib_modularity = evaluation.modularity_overlap(snapshot, communities).score

            intermediate = IntermediateResults(
                runtime=elapsed,
                cdlib_modularity_overlap=cdlib_modularity,
                customize_q0_overlap=q0_modularity,
                affected_nodes=snapshot.number_of_nodes(),
                num_communities=len(communities.communities),
            )
            results[algo_name].update_intermediate_results(intermediate)

    return results

def run_dynamic_benchmark(
    temporal_graph: TemporalGraph,
    algorithm_names: list[str] | None = None,
) -> dict[str, MethodDynamicResults]:
    """
    Run dynamic community detection algorithms on a temporal graph.

    Args:
        temporal_graph: TemporalGraph with snapshots to benchmark
        algorithm_names: List of algorithm names to run. Defaults to ["tiles"]

    Returns:
        Dictionary mapping algorithm name to MethodDynamicResults
    """
    ALGORITHMS_OVERLAPPING = {
        "tiles": {
            "func": dynamic_methods.Tiles(obs=1),
            "params": {},
        }
    }
    if algorithm_names is None:
        algorithm_names = ALGORITHMS_OVERLAPPING.keys()

    results: dict[str, MethodDynamicResults] = {
        name: MethodDynamicResults() for name in algorithm_names
    }

    algorithms_bar = tqdm(algorithm_names, total=len(algorithm_names), desc="Algorithms", leave=False)
    for algo_name in algorithms_bar:
        algo_func = ALGORITHMS_OVERLAPPING[algo_name]["func"]
        algo_params = ALGORITHMS_OVERLAPPING[algo_name]["params"]
        res = algo_func(temporal_graph, **algo_params)
        results[algo_name] = res
    return results
