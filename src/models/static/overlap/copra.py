"""COPRA: Community Overlap Propagation in Real Applications."""

import collections
import random
from typing import List

import networkx as nx
import numpy as np
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class COPRA(CommunityDetectionAlgorithm):
    """
    Detect overlapping communities using label propagation algorithm.

    The algorithm propagates community labels across nodes, allowing each node
    to belong to multiple communities based on membership strength.
    """

    def __init__(self, iterations: int = 10, max_communities: int = 5, seed: int = 42):
        """
        Initialize COPRA parameters.

        Args:
            iterations: Number of propagation iterations (default 10)
            max_communities: Maximum number of communities a node can belong to (default 5)
            seed: Random seed for reproducibility (default 42)
        """
        self.iterations = iterations
        self.max_communities = max_communities
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def fit_predict(self, G: nx.Graph) -> List[List]:
        """
        Detect overlapping communities using label propagation.

        Args:
            G: NetworkX graph

        Returns:
            List of communities (each community is a list of node IDs)
        """
        node_labels = {node: {node: 1} for node in G.nodes()}

        for _ in range(self.iterations):
            visit_order = list(G.nodes())
            np.random.shuffle(visit_order)

            for node in visit_order:
                label_count = 0
                accumulated_labels = {}
                neighbor_count = len(G[node])

                for neighbor in G.neighbors(node):
                    neighbor_contribution = {
                        label: value / neighbor_count
                        for label, value in node_labels[neighbor].items()
                    }
                    accumulated_labels = dict(
                        collections.Counter(neighbor_contribution)
                        + collections.Counter(accumulated_labels)
                    )

                label_count = len(accumulated_labels)
                accumulated_labels_backup = accumulated_labels.copy()

                for label, value in list(accumulated_labels.items()):
                    if value < 1 / self.max_communities:
                        del accumulated_labels[label]
                        label_count -= 1

                if label_count == 0:
                    selected_label = random.sample(list(accumulated_labels_backup.keys()), 1)
                    accumulated_labels = {selected_label[0]: 1}
                else:
                    label_sum = sum(accumulated_labels.values())
                    accumulated_labels = {
                        label: value / label_sum
                        for label, value in accumulated_labels.items()
                    }

                node_labels[node] = accumulated_labels

        communities = collections.defaultdict(list)

        for node, labels in node_labels.items():
            for label in labels.keys():
                communities[label].append(node)

        return list(communities.values())

    def _process_snapshot(self, G: nx.Graph) -> NodeClustering:
        """
        Detect communities and return as NodeClustering object.

        Args:
            G: NetworkX graph

        Returns:
            NodeClustering object with detected communities
        """
        communities = self.fit_predict(G)
        return NodeClustering(communities=communities, graph=G)

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """Run COPRA on each snapshot of the temporal graph."""
        results = []
        for snapshot in tg.iter_snapshots():
            results.append(self._process_snapshot(snapshot))
        return results


def copra(G: nx.Graph, iterations: int = 10, max_communities: int = 5, seed: int = 42) -> NodeClustering:
    """
    Convenience function for COPRA community detection on a single graph.

    Args:
        G: NetworkX graph
        iterations: Number of propagation iterations (default 10)
        max_communities: Maximum number of communities per node (default 5)
        seed: Random seed for reproducibility (default 42)

    Returns:
        NodeClustering object with detected communities
    """
    model = COPRA(iterations=iterations, max_communities=max_communities, seed=seed)
    return model._process_snapshot(G)
