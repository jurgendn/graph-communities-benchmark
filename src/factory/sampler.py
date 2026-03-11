from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


class SelectiveSampler:
    def __init__(
        self,
        G: nx.Graph,
        communities: Dict[int, int],
        num_communities_range: Tuple[int, int] = (1, 5),
    ):
        self.G = G
        self.communities = communities
        self.num_communities_range = num_communities_range

        self.num_static_edges = G.number_of_edges()

    def update_communities(self, communities: Dict[int, int]):
        self.communities = communities

    def update_graph(self, G: nx.Graph):
        self.G = G
        self.num_static_edges = G.number_of_edges()

    def __is_eligible_to_remove(
        self, u: int, v: int, target_communities: List[int]
    ) -> bool:
        """Check if an edge can be removed based on community membership."""
        community_u = self.communities[u]
        community_v = self.communities[v]

        is_eligible = community_u == community_v and community_u in target_communities

        return is_eligible

    def sample(self, num_samples: int) -> List[Tuple]:
        """Sampling edges from the graph based on community membership.
        Randomly selects edges within community among the specified number of communities.

        Args:
            num_samples (int): _description_
            num_communities (int): _description_

        Returns:
            List[Tuple]: _description_
        """
        shuffled_edges = np.random.permutation(list(self.G.edges()))
        communities_list = list(self.communities.values())
        num_communities = np.random.randint(
            self.num_communities_range[0], self.num_communities_range[1] + 1
        )

        selected_communities = np.random.choice(
            communities_list, size=num_communities, replace=False
        )
        sampled_edges = []
        for edge in shuffled_edges:
            u, v = edge
            if self.__is_eligible_to_remove(u, v, selected_communities) is True:
                sampled_edges.append(edge)
                if len(sampled_edges) >= num_samples:
                    break
        return sampled_edges
