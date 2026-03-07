from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
from cdlib import NodeClustering
from networkx.algorithms.community import modularity

from src.factory.sampler import SelectiveSampler
from src.models.utils.helpers import convert_dict_communities_to_cdlib


class LouvainMixin:
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
        self.graph = graph.copy()
        self.sampler_type = sampler_type
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        if isinstance(graph, nx.DiGraph):
            self.graph = graph.to_undirected()

        self.nodes = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        if initial_communities is not None:
            communities_dict = initial_communities
        else:
            # Initialize communities using networkx louvain_communities
            communities_list = nx.algorithms.community.louvain_communities(self.graph, seed=42)
            communities_dict = {}
            for community_id, community_nodes in enumerate(communities_list):
                for node in community_nodes:
                    communities_dict[node] = community_id
        self.communities_dict = communities_dict
        self.community = np.zeros(len(self.nodes), dtype=int)
        for node, community_id in communities_dict.items():
            if node in self.node_to_idx:
                node_idx = self.node_to_idx[node]
                self.community[node_idx] = community_id

        self.weighted_degree = self._calculate_weighted_degrees()
        self.community_weights = self._calculate_community_weights()
        self.total_edge_weight = sum(self.weighted_degree) / 2

        self.affected = np.zeros(len(self.nodes), dtype=bool)

        self.sampler = SelectiveSampler(
            G=self.graph,
            communities=communities_dict,
            num_communities_range=num_communities_range,
        )

    def _calculate_weighted_degrees(self) -> np.ndarray:
        degrees = np.zeros(len(self.nodes), dtype=float)
        for i, node in enumerate(self.nodes):
            degree = 0.0
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get("weight", 1.0)
                degree += weight
            degrees[i] = degree
        return degrees

    def _calculate_community_weights(self) -> Dict[int, float]:
        from collections import defaultdict

        weights = defaultdict(float)
        for i, node in enumerate(self.nodes):
            community_id = self.community[i]
            weights[community_id] += self.weighted_degree[i]
        return dict(weights)

    def _get_neighbor_communities(self, node_idx: int) -> Dict[int, float]:
        from collections import defaultdict

        node = self.nodes[node_idx]
        neighbor_communities = defaultdict(float)

        for neighbor in self.graph.neighbors(node):
            if neighbor in self.node_to_idx:
                neighbor_idx = self.node_to_idx[neighbor]
                neighbor_community = self.community[neighbor_idx]
                weight = self.graph[node][neighbor].get("weight", 1.0)
                neighbor_communities[neighbor_community] += weight

        return neighbor_communities

    def _calculate_delta_modularity(
        self, node_idx: int, target_community: int
    ) -> float:
        current_community = self.community[node_idx]

        if current_community == target_community:
            return 0.0

        neighbor_communities = self._get_neighbor_communities(node_idx)
        k_i_to_c = neighbor_communities.get(target_community, 0.0)
        k_i_to_d = neighbor_communities.get(current_community, 0.0)

        k_i = self.weighted_degree[node_idx]
        sigma_c = self.community_weights.get(target_community, 0.0)
        sigma_d = self.community_weights.get(current_community, 0.0)

        delta_q = (1.0 / self.total_edge_weight) * (k_i_to_c - k_i_to_d) - (
            k_i / (2.0 * self.total_edge_weight**2)
        ) * (k_i + sigma_c - sigma_d)

        return delta_q

    def _mark_neighbors_affected(self, node_idx: int) -> None:
        node = self.nodes[node_idx]
        for neighbor in self.graph.neighbors(node):
            if neighbor in self.node_to_idx:
                neighbor_idx = self.node_to_idx[neighbor]
                self.affected[neighbor_idx] = True

    def _add_new_nodes(self, new_nodes: List[Any]) -> None:
        for node in new_nodes:
            if node not in self.node_to_idx:
                self.graph.add_node(node)
                node_idx = len(self.nodes)
                self.nodes.append(node)
                self.node_to_idx[node] = node_idx
                self.community = np.append(self.community, node_idx)
                self.weighted_degree = np.append(self.weighted_degree, 0.0)
                self.affected = np.append(self.affected, False)
                self.community_weights[node_idx] = 0.0

    def get_modularity(self) -> float:
        communities_dict = {
            self.nodes[i]: self.community[i] for i in range(len(self.nodes))
        }
        # Convert communities_dict to list of sets for modularity calculation
        communities = {}
        for node, community_id in communities_dict.items():
            if community_id not in communities:
                communities[community_id] = set()
            communities[community_id].add(node)
        communities_list = list(communities.values())
        return modularity(self.graph, communities_list)

    def get_communities_dict(self) -> Dict[int, set]:
        """Return communities as a dictionary mapping community_id -> set(nodes)."""
        from collections import defaultdict

        communities = defaultdict(set)
        for i, node in enumerate(self.nodes):
            communities[self.community[i]].add(node)
        return dict(communities)

    def get_communities(self) -> NodeClustering:
        """Return communities as a NodeClustering object."""
        communities_dict = self.get_communities_dict()
        return convert_dict_communities_to_cdlib(self.graph, communities_dict)

    def get_affected_nodes(self) -> List[Any]:
        return [self.nodes[i] for i in range(len(self.nodes)) if self.affected[i]]
