import time

from cdlib import algorithms, evaluation
from dynetx import DynGraph

from src.factory.communities import IntermediateResults, MethodDynamicResults
from src.factory.factory import TemporalGraph


class Tiles:
    def __init__(self):
        pass

    def run(self, tg: TemporalGraph) -> MethodDynamicResults:
        """
        Apply the TILES algorithm to a temporal graph.

        Args:
            tg: A dynamic graph (DynGraph) representing the temporal network.

        Returns:
            A list of TemporalClustering objects, one for each snapshot.
        """
        dgraph = DynGraph()
        for t, snapshot in enumerate(tg.iter_snapshots()):
            dgraph.add_interactions_from(list(snapshot.edges()), t=t)
        start_time = time.perf_counter()
        res = algorithms.tiles(dg=dgraph, obs=1)
        elapsed = time.perf_counter() - start_time
        results = MethodDynamicResults()
        for snapshot, community in zip(tg.iter_snapshots(), res.clusterings.values()):
            q = evaluation.modularity_overlap(snapshot, community).score
            current_res = IntermediateResults(
                runtime=elapsed,
                modularity=q,
                affected_nodes=snapshot.number_of_nodes(),
                num_communities=len(community.communities),
            )
            results.update_intermediate_results(current_res)
        return results

    def __call__(self, tg: TemporalGraph):
        res = self.run(tg)
        return res