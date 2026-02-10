from typing import Optional
from cdlib import NodeClustering
import networkx as nx
import torch


class NDOCDTorch:
    """
    GPU-accelerated implementation of Overlapping Community Detection based on 
    Network Decomposition (NDOCD) using PyTorch.
    Ref: Ding et al. (2016)
    
    This version supports GPU acceleration for faster computation on large graphs.
    """
    def __init__(self, beta: float = 0.3, 
                 threshold_js: float = 0.4, threshold_md: float = 0.5, 
                 min_community_size: int = 3,
                 device: Optional[str] = None):
        """
        Initialize the NDOCD parameters.
        
        Args:
            beta: Parameter for CNFV (default 0.3).
            threshold_js: Threshold for Joint Strength (default 0.4).
            threshold_md: Threshold for Membership Degree (default 0.5).
            min_community_size: Minimum size to retain a community.
            device: Torch device ('cuda', 'cpu', or None for auto-detect).
        """
        self.beta = beta
        self.threshold_js = threshold_js
        self.threshold_md = threshold_md
        self.min_community_size = min_community_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.G_curr: Optional[nx.Graph] = None
        self.adj_matrix: Optional[torch.Tensor] = None
        self.node_to_idx: Optional[dict] = None
        self.idx_to_node: Optional[dict] = None
        self.active_nodes: Optional[torch.Tensor] = None

    def _build_adjacency_matrix(self, G: nx.Graph) -> None:
        """Build adjacency matrix and node mappings from NetworkX graph."""
        nodes = sorted(list(G.nodes()))
        num_nodes = len(nodes)
        
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Build sparse adjacency matrix
        rows, cols = [], []
        for u, v in G.edges():
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
        
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
            values = torch.ones(len(rows), dtype=torch.float32, device=self.device)
            self.adj_matrix = torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes), device=self.device
            ).coalesce()
        else:
            self.adj_matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
                (num_nodes, num_nodes), device=self.device
            )
        
        # Initialize all nodes as active
        self.active_nodes = torch.ones(num_nodes, dtype=torch.bool, device=self.device)

    def _get_dense_adj_for_active(self) -> torch.Tensor:
        """Get dense adjacency matrix for active nodes."""
        if self.adj_matrix is None:
            return torch.tensor([], device=self.device)
        
        active_indices = torch.where(self.active_nodes)[0]
        n_active = len(active_indices)
        
        if n_active == 0:
            return torch.zeros((0, 0), device=self.device)
        
        # Convert sparse to dense
        dense = self.adj_matrix.to_dense()
        
        # Extract submatrix for active nodes
        submatrix = dense[active_indices][:, active_indices]
        return submatrix

    def _calculate_degrees(self, adj_dense: torch.Tensor) -> torch.Tensor:
        """Calculate node degrees from adjacency matrix."""
        return adj_dense.sum(dim=1)

    def _calculate_clustering_coefficients(self, adj_dense: torch.Tensor) -> torch.Tensor:
        """
        Calculate clustering coefficients for all active nodes.
        C_i = (number of triangles connected to i) / (k_i * (k_i - 1) / 2)
        """
        n = adj_dense.shape[0]
        if n == 0:
            return torch.tensor([], device=self.device)
        
        # For each node, count triangles: A^3 gives paths of length 3
        # Diagonal elements divided by 2 give triangles
        A_squared = torch.mm(adj_dense, adj_dense)
        triangles = torch.diagonal(torch.mm(A_squared, adj_dense)) / 2.0
        
        degrees = adj_dense.sum(dim=1)
        possible_triangles = degrees * (degrees - 1) / 2.0
        
        # Avoid division by zero
        clustering = torch.zeros_like(degrees, dtype=torch.float32)
        mask = possible_triangles > 0
        clustering[mask] = triangles[mask] / possible_triangles[mask]
        
        return clustering

    def _calculate_cnfv(self, active_indices: torch.Tensor, adj_dense: torch.Tensor) -> torch.Tensor:
        """Calculate Comprehensive Network Feature Value (CNFV) for active nodes."""
        degrees = self._calculate_degrees(adj_dense)
        clustering = self._calculate_clustering_coefficients(adj_dense)
        
        n_active = len(active_indices)
        cnfv = (self.beta * clustering) + ((1 - self.beta) * (degrees / n_active))
        
        # Set CNFV to -1 for isolated nodes
        cnfv[degrees == 0] = -1.0
        
        return cnfv

    def _get_neighbors_tensor(self, node_idx: int, adj_dense: torch.Tensor) -> torch.Tensor:
        """Get neighbor indices for a given active node."""
        neighbors = torch.where(adj_dense[node_idx] > 0)[0]
        return neighbors

    def _get_centered_clique(self, active_indices: torch.Tensor, 
                             adj_dense: torch.Tensor):
        """Step 1: Seed Selection using CNFV."""
        if adj_dense.shape[0] == 0:
            return None
        
        # Calculate CNFV for all active nodes
        cnfv = self._calculate_cnfv(active_indices, adj_dense)
        
        # Find node with highest CNFV
        valid_mask = cnfv > -1.0
        if not valid_mask.any():
            return None
        
        best_local_idx = torch.argmax(cnfv[valid_mask])
        best_local_idx = torch.where(valid_mask)[0][best_local_idx]
        
        if adj_dense[best_local_idx].sum() == 0:
            return None
        
        # Build clique centered in best_node
        clique = {best_local_idx.item()}
        neighbors = self._get_neighbors_tensor(best_local_idx, adj_dense)
        
        # Sort neighbors by degree
        neighbor_degrees = adj_dense[neighbors].sum(dim=1)
        sorted_neighbors = neighbors[torch.argsort(neighbor_degrees, descending=True)]
        
        for neighbor_idx in sorted_neighbors:
            neighbor_idx = neighbor_idx.item()
            # Check if neighbor is connected to all members in clique
            is_connected_to_all = True
            for member_idx in clique:
                if adj_dense[neighbor_idx, member_idx] == 0:
                    is_connected_to_all = False
                    break
            
            if is_connected_to_all:
                clique.add(neighbor_idx)
        
        return clique

    def _expand_seed(self, seed, adj_dense: torch.Tensor):
        """Step 2: Seed Expansion using Joint Strength (JS) and Membership Degree (MD)."""
        community = set(seed)
        
        while True:
            added = False
            shell = set()
            
            # Find shell nodes (neighbors of community not in community)
            for node_idx in community:
                neighbors = self._get_neighbors_tensor(node_idx, adj_dense)
                for neighbor_idx in neighbors:
                    neighbor_idx = neighbor_idx.item()
                    if neighbor_idx not in community:
                        shell.add(neighbor_idx)
            
            n_k = len(community)
            candidates = []
            
            for node_idx in shell:
                # Count connections to community
                community_list = list(community)
                connections = sum(1 for member in community_list 
                                if adj_dense[node_idx, member] > 0)
                
                degree = adj_dense[node_idx].sum().item()
                
                if degree == 0:
                    continue
                
                js = connections / n_k
                md = connections / degree
                
                if js > self.threshold_js or md > self.threshold_md:
                    candidates.append(node_idx)
            
            # Add candidates to community
            for cand_idx in candidates:
                if cand_idx not in community:
                    community.add(cand_idx)
                    added = True
            
            if not added:
                break
        
        return community

    def _remove_community_edges(self, community) -> None:
        """Remove edges within the detected community from the graph."""
        if not community or self.adj_matrix is None:
            return
        
        # Create mask for edges to remove
        indices = self.adj_matrix.indices()
        
        # Find edges where both endpoints are in community
        rows, cols = indices[0], indices[1]
        mask = torch.tensor(
            [row.item() in community and col.item() in community 
             for row, col in zip(rows, cols)],
            device=self.device
        )
        
        # Keep only edges not in community
        new_indices = indices[:, ~mask]
        new_values = self.adj_matrix.values()[~mask]
        
        if new_indices.shape[1] > 0:
            self.adj_matrix = torch.sparse_coo_tensor(
                new_indices, new_values, self.adj_matrix.shape, device=self.device
            ).coalesce()
        else:
            self.adj_matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
                self.adj_matrix.shape, device=self.device
            )

    def execute(self, G: nx.Graph) -> NodeClustering:
        """
        Executes the main NDOCD loop with GPU acceleration.
        
        Args:
            G: The input NetworkX graph.
            
        Returns:
            NodeClustering
        """
        # Initialize
        self._build_adjacency_matrix(G)
        final_communities = []
        
        # Loop until no edges remain
        while self.adj_matrix._nnz() > 0:
            # Get active nodes and their dense adjacency matrix
            active_indices = torch.where(self.active_nodes)[0]
            adj_dense = self._get_dense_adj_for_active()
            
            if adj_dense.shape[0] == 0:
                break
            
            # 1. Seed Selection
            seed = self._get_centered_clique(active_indices, adj_dense)
            
            if not seed or len(seed) < 2:
                break
            
            # Convert local indices to global node indices
            # seed_global = {self.idx_to_node[active_indices[local_idx].item()] 
            #               for local_idx in seed}
            
            # 2. Seed Expansion (work with local indices)
            community_local = self._expand_seed(seed, adj_dense)
            
            if len(community_local) >= self.min_community_size:
                # Convert back to global node indices
                community_global = {self.idx_to_node[active_indices[local_idx].item()] 
                                  for local_idx in community_local}
                final_communities.append(community_global)
            
            # 3. Network Decomposition
            self._remove_community_edges(community_local)
        
        # Clean up GPU memory
        self.adj_matrix = None
        self.active_nodes = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Construct and return the NodeCluster
        communities_as_lists = [list(comm) for comm in final_communities]
        
        node_clustering = NodeClustering(
            communities=communities_as_lists,
            graph=G,
            method_name="NDOCD-Torch",
            method_parameters={
                "beta": self.beta,
                "threshold_js": self.threshold_js,
                "threshold_md": self.threshold_md,
                "min_community_size": self.min_community_size,
                "device": str(self.device),
            },
            overlap=True,
        )
        return node_clustering

    def __call__(self, G: nx.Graph) -> NodeClustering:
        return self.execute(G)
