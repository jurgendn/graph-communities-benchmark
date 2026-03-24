"""Static Louvain: crisp community detection run independently per snapshot."""

from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.core.temporal_graph import TemporalGraph


@register(
    name="static_louvain",
    algo_type="static",
    clustering_type="crisp",
    default_params={"seed": 42},
    description="NetworkX Louvain run independently on each snapshot",
)
class StaticLouvain(CommunityDetectionAlgorithm):
    """
    Crisp (non-overlapping) community detection using NetworkX Louvain,
    run independently per snapshot.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _process_snapshot(self, G: nx.Graph) -> NodeClustering:
        """Run Louvain on a single graph snapshot."""
        communities = nx.algorithms.community.louvain_communities(G, seed=self.seed)
        communities_list = [list(c) for c in communities]
        return NodeClustering(
            communities=communities_list,
            graph=G,
            method_name="StaticLouvain",
        )

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """Run Louvain on each snapshot of the temporal graph."""
        results = []
        for snapshot in tg.iter_snapshots():
            results.append(self._process_snapshot(snapshot))
        return results
