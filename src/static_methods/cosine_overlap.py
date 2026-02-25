"""CosineOverlap: Overlapping community detection using cosine similarity."""

import networkx as nx
import networkx.algorithms.community as nxcom
import numpy as np
from cdlib import NodeClustering
from numpy import linalg as LA


class CosineOverlap:
    """
    Detect overlapping communities using cosine similarity in embedding space.
    
    The algorithm:
    1. Detects initial communities using Louvain method
    2. Embeds nodes into a vector space using random walk transition matrix
    3. Expands communities by adding nodes with high cosine similarity to community centers
    """
    
    def __init__(
        self,
        theta: float = 0.85,
        walk_power: int = 3,
        seed: int = 123
    ):
        """
        Initialize CosineOverlap parameters.
        
        Args:
            theta: Cosine similarity threshold for adding nodes to communities (default 0.85)
            walk_power: Power to raise the transition matrix (default 3)
            seed: Random seed for Louvain initial clustering (default 123)
        """
        self.theta = theta
        self.walk_power = walk_power
        self.seed = seed
    
    def _compute_degree_matrix_sqrt_inv(self, degree_matrix: np.ndarray) -> np.ndarray:
        """
        Compute diagonal matrix with elements D[i][i]^(-1/2).
        
        Args:
            degree_matrix: Degree matrix D
            
        Returns:
            Diagonal matrix with D[i][i]^(-1/2) on diagonal
        """
        n = degree_matrix.shape[0]
        result = np.zeros((n, n))
        for i in range(n):
            if degree_matrix[i, i] > 0:
                result[i, i] = 1.0 / np.sqrt(degree_matrix[i, i])
        return result
    
    def _compute_community_center(self, community: list, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a community in embedding space.
        
        Args:
            community: List of node indices in the community
            embeddings: Node embedding matrix
            
        Returns:
            Centroid vector of the community
        """
        if not community:
            return np.zeros(embeddings.shape[1])
        
        community_embeddings = embeddings[community]
        return np.mean(community_embeddings, axis=0)
    
    def fit_predict(self, G: nx.Graph):
        """
        Detect overlapping communities in the graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            List of communities (each community is a list of node IDs)
        """
        # Store node list to map indices to actual node IDs
        self.node_list = list(G.nodes())
        num_nodes = len(self.node_list)
        
        # Build adjacency and degree matrices
        adj_matrix = nx.to_numpy_array(G, nodelist=self.node_list, dtype=float)
        degree_matrix = np.diag(adj_matrix.sum(axis=1))
        
        # Compute transition matrix P = D^(-1) @ A
        degree_inv = np.diag(1.0 / np.maximum(np.diag(degree_matrix), 1e-10))
        transition_matrix = degree_inv @ adj_matrix
        
        # Raise transition matrix to power
        transition_powered = np.linalg.matrix_power(transition_matrix, self.walk_power)
        
        # Compute node embeddings: D^(-1/2) @ P^t
        degree_sqrt_inv = self._compute_degree_matrix_sqrt_inv(degree_matrix)
        embeddings = np.array([
            (degree_sqrt_inv @ transition_powered[i].T).flatten()
            for i in range(num_nodes)
        ])
        
        # Get initial communities using Louvain
        initial_communities = nxcom.louvain_communities(G, seed=self.seed)
        
        # Expand communities using cosine similarity
        overlapping_communities = []
        for community in initial_communities:
            # Convert community to list of indices
            community_indices = [self.node_list.index(node) for node in community]
            expanded_community = set(community)
            
            # Compute community center in embedding space
            center = self._compute_community_center(community_indices, embeddings)
            center_norm = LA.norm(center)
            
            if center_norm == 0:
                overlapping_communities.append(list(expanded_community))
                continue
            
            # Check all nodes for potential membership
            for node_idx in range(num_nodes):
                node_embedding = embeddings[node_idx]
                node_norm = LA.norm(node_embedding)
                
                if node_norm == 0:
                    continue
                
                # Compute cosine similarity
                cosine_similarity = np.dot(node_embedding, center) / (node_norm * center_norm)
                
                # Add node if similarity exceeds threshold
                if cosine_similarity > self.theta:
                    actual_node = self.node_list[node_idx]
                    expanded_community.add(actual_node)
            
            overlapping_communities.append(list(expanded_community))
        
        return overlapping_communities
    
    def __call__(self, G: nx.Graph) -> NodeClustering:
        """
        Detect communities and return as NodeClustering object.
        
        Args:
            G: NetworkX graph
            
        Returns:
            NodeClustering object with detected communities
        """
        communities = self.fit_predict(G)
        return NodeClustering(communities=communities, graph=G)


def cosine_overlap(G: nx.Graph, theta: float = 0.85, walk_power: int = 3, seed: int = 123) -> NodeClustering:
    """
    Convenience function for CosineOverlap community detection.
    
    Args:
        G: NetworkX graph
        theta: Cosine similarity threshold (default 0.85)
        walk_power: Power to raise the transition matrix (default 3)
        seed: Random seed for Louvain (default 123)
        
    Returns:
        NodeClustering object with detected communities
    """
    model = CosineOverlap(theta=theta, walk_power=walk_power, seed=seed)
    return model(G)