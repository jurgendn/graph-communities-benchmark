"""BigClam: Overlapping community detection algorithm."""

import math

import networkx as nx
import numpy as np
from cdlib import NodeClustering


class BigClam:
    """Detect overlapping communities using gradient ascent on affiliation matrix."""

    def __init__(
        self,
        num_communities: int = 4,
        iterations: int = 100,
        learning_rate: float = 0.005,
    ):
        self.k = num_communities
        self.iterations = iterations
        self.lr = learning_rate
        self.F = None
    
    def fit(self, G: nx.Graph):
        """Learn community affiliations from graph."""
        # Store node list to map indices to actual node IDs
        self.node_list = list(G.nodes())
        adj = nx.to_numpy_array(G, nodelist=self.node_list)
        n = adj.shape[0]
        self.F = np.random.rand(n, self.k)
        
        # Gradient ascent
        for _ in range(self.iterations):
            for i in range(n):
                neighbors = np.where(adj[i] == 1)[0]
                non_neighbors = np.where(adj[i] == 0)[0]
                
                # Gradient from neighbors
                grad = np.zeros(self.k)
                for j in neighbors:
                    dot = self.F[j] @ self.F[i]
                    grad += self.F[j] * np.exp(-dot) / (1 - np.exp(-dot))
                
                # Gradient from non-neighbors
                grad -= np.sum(self.F[non_neighbors], axis=0)
                
                # Update
                self.F[i] += self.lr * grad
                self.F[i] = np.maximum(0.001, self.F[i])
        
        return self
    
    def get_communities(self, G: nx.Graph):
        """Extract communities using density-based threshold."""
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # Compute threshold from graph density
        density = 2 * m / (n * (n - 1)) if n > 1 else 0
        threshold = math.sqrt(-math.log(1 - density)) if density < 1 else 0
        
        # Assign nodes to communities (map indices back to actual node IDs)
        communities = [[] for _ in range(self.k)]
        for idx, affinities in enumerate(self.F): # pyright: ignore[reportArgumentType]
            actual_node = self.node_list[idx]  # Map index to actual node ID
            for comm in np.where(affinities > threshold)[0]:
                communities[comm].append(actual_node)
        
        return [c for c in communities if c]  # Remove empty communities

    def fit_predict(self, G: nx.Graph):
        """Fit and return communities in one step."""
        return self.fit(G).get_communities(G)

    def __call__(self, G: nx.Graph) -> NodeClustering:
        communities = self.fit_predict(G)
        return NodeClustering(communities=communities, graph=G)


def big_clam(G: nx.Graph, num_communities: int = 10, iterations: int = 100, learning_rate: float = 0.005) -> NodeClustering:
    """Convenience function for BigClam."""
    model = BigClam(num_communities=num_communities, iterations=iterations, learning_rate=learning_rate)
    return model(G)