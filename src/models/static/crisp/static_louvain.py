from time import time
from typing import Any, Dict, List, Literal, Optional, Text, Tuple

import networkx as nx
from cdlib import NodeClustering
from networkx.algorithms.community import modularity

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.communities import IntermediateResults, MethodDynamicResults
from src.factory.factory import TemporalGraph
from src.models.common.louvain_base import LouvainMixin


class StaticLouvain(CommunityDetectionAlgorithm, LouvainMixin):
    def __init__(
        self,
        graph: nx.Graph,
        initial_communities: Optional[Dict[Any, int]] = None,
        sampler_type: Literal["selective", "full"] = "selective",
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        verbose: bool = True,
        num_communities_range: Tuple[int, int] = (1, 5),
    ) -> None:
        super().__init__(
            graph=graph,
            initial_communities=initial_communities,
            sampler_type=sampler_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            num_communities_range=num_communities_range,
        )
        self.__shortname__ = "Static Louvain"

    def apply_batch_update(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> None:
        if edge_deletions:
            for edge in edge_deletions:
                if len(edge) == 2:
                    node1, node2 = edge
                else:
                    node1, node2 = edge[0], edge[1]
                if self.graph.has_edge(node1, node2):
                    if node1 in self.node_to_idx and node2 in self.node_to_idx:
                        weight = self.graph[node1][node2].get("weight", 1.0)
                        self.graph.remove_edge(node1, node2)
                    else:
                        self.graph.remove_edge(node1, node2)
        if edge_insertions:
            for edge in edge_insertions:
                if len(edge) == 3:
                    node1, node2, weight = edge
                else:
                    node1, node2 = edge[0], edge[1]
                    weight = 1.0
                new_nodes = []
                for node in [node1, node2]:
                    if node not in self.node_to_idx:
                        new_nodes.append(node)
                if new_nodes:
                    self._add_new_nodes(new_nodes)
                self.graph.add_edge(node1, node2, weight=weight)

    def run(
        self,
        edge_deletions: List[Tuple],
        edge_insertions: List[Tuple],
    ) -> NodeClustering:
        if self.sampler_type == "selective":
            edge_deletions = self.sampler.sample(
                num_samples=len(edge_deletions),
            )
        if edge_deletions or edge_insertions:
            self.apply_batch_update(edge_deletions, edge_insertions)
        start_time = time()
        communities = nx.algorithms.community.louvain_communities(self.graph, seed=42)
        for community_id, community_nodes in enumerate(communities):  # type: ignore
            for node in community_nodes:
                self.community[self.node_to_idx[node]] = community_id
        self.sampler.update_communities(
            {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        )
        self.sampler.update_graph(self.graph)
        runtime = time() - start_time
        # runtime is intentionally computed here to allow metric helpers to reuse it if needed
        return self.get_communities()

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run Static Louvain on each snapshot of a temporal graph.

        Args:
            tg: TemporalGraph with snapshots.

        Returns:
            List[NodeClustering], one per snapshot, in same order as
            ``tg.iter_snapshots()``.
        """
        results = []
        # Snapshot t = 0: base graph (no edge changes yet)
        base_clustering = self.get_communities()
        results.append(base_clustering)

        # Snapshots t = 1...T: apply each batch of changes
        for step in tg.steps:
            # Prepare edge insertions with normalized weight format
            edge_insertions = []
            for edge in step.insertions:
                if len(edge) == 3:
                    u, v, data = edge
                    weight = data.get("weight", 1.0) if isinstance(data, dict) else float(data)
                else:
                    u, v = edge[0], edge[1]
                    weight = 1.0
                edge_insertions.append((u, v, weight))

            # Apply changes using existing run() method
            self.run(step.deletions, edge_insertions)

            # Get clustering for this snapshot
            clustering = self.get_communities()
            results.append(clustering)

        return results


class StaticLouvainWrapper(CommunityDetectionAlgorithm):
    """
    Implements CommunityDetectionAlgorithm for crisp (non-overlapping)
    community detection using NetworkX Louvain run independently per snapshot.

    This class conforms to the unified interface and can be registered in
    ``config/algorithms.yaml`` under ``type: static``.
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
