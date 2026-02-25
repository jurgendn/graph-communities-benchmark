"""
Optimized Overlapping Modularity Implementation with Numba Acceleration
========================================================================

Formula (5.1):
Q = (1/2m) * sum_{C_j} sum_{u,v in C_j} (A_uv - d_u*d_v/2m) * f(alpha_uC, alpha_vC)

where f(alpha_u, alpha_v) = (alpha_u + alpha_v) / 2

Key property: sum_{C_j} alpha_{u,C_j} = 1 for all nodes u
This ensures Q is bounded in [-1, 1]
"""

from typing import List, Set, Union

import networkx as nx
import numpy as np
from cdlib import NodeClustering, evaluation
from scipy.sparse import csr_matrix

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available. Install with: pip install numba")

Communities = Union[List[Set], NodeClustering]

def _extract_communities(communities: Communities) -> List[Set]:
    """Extract community list from various input formats."""
    if hasattr(communities, 'communities'):
        return [set(comm) for comm in communities.communities]
    return [set(comm) for comm in communities]


if HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _compute_alpha_matrix_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        comm_masks: np.ndarray,
        node_in_comm: np.ndarray,
        n_nodes: int,
        n_comms: int
    ) -> np.ndarray:
        """
        Compute normalized alpha matrix.
        
        alpha[u, C_j] = d_{u,C_j} / sum_{C_k} d_{u,C_k}
        
        Edge cases:
        - If node u has no edges to any community, distribute alpha uniformly
        across communities that u belongs to
        - If node u belongs to no communities, alpha stays 0
        
        Guarantees: sum_{C_j} alpha[u, C_j] = 1 for all u in at least one community
        """
        alpha = np.zeros((n_nodes, n_comms), dtype=np.float64)
        
        for u in prange(n_nodes):
            start, end = indptr[u], indptr[u + 1]
            
            # Compute d_{u,C} for each community
            for c in range(n_comms):
                d_u_c = 0.0
                for j in range(start, end):
                    v = indices[j]
                    if comm_masks[c, v]:
                        d_u_c += 1.0
                alpha[u, c] = d_u_c
            
            # Normalize so sum over communities = 1
            total = 0.0
            for c in range(n_comms):
                total += alpha[u, c]
            
            if total > 0:
                # Normal case: normalize by total edges to communities
                for c in range(n_comms):
                    alpha[u, c] /= total
            else:
                # Edge case: node has no edges to any community members
                # Distribute uniformly across communities node belongs to
                n_memberships = 0
                for c in range(n_comms):
                    if node_in_comm[u, c]:
                        n_memberships += 1
                
                if n_memberships > 0:
                    uniform_alpha = 1.0 / n_memberships
                    for c in range(n_comms):
                        if node_in_comm[u, c]:
                            alpha[u, c] = uniform_alpha
        
        return alpha

    @njit(cache=True, parallel=True)
    def _compute_Q_for_community_numba(
        indptr_w: np.ndarray,
        indices_w: np.ndarray,
        data_w: np.ndarray,
        degrees: np.ndarray,
        comm_indices: np.ndarray,
        alpha: np.ndarray,
        two_m: float
    ) -> float:
        """
        Compute Q contribution for a single community.
        """
        n = len(comm_indices)
        if n == 0:
            return 0.0
        
        row_contributions = np.zeros(n, dtype=np.float64)
        
        for i in prange(n):
            u = comm_indices[i]
            d_u = degrees[u]
            alpha_u = alpha[i]
            row_start, row_end = indptr_w[u], indptr_w[u + 1]
            
            row_sum = 0.0
            for j in range(n):
                v = comm_indices[j]
                d_v = degrees[v]
                alpha_v = alpha[j]
                
                # Find A_uv
                A_uv = 0.0
                for k in range(row_start, row_end):
                    if indices_w[k] == v:
                        A_uv = data_w[k]
                        break
                
                null_model = (d_u * d_v) / two_m
                f_alpha = (alpha_u + alpha_v) / 2.0
                row_sum += (A_uv - null_model) * f_alpha
            
            row_contributions[i] = row_sum
        
        return row_contributions.sum()


def overlapping_modularity_q0(G: nx.Graph, communities: Communities) -> float:
    comm_list = _extract_communities(communities)
    
    if not comm_list:
        return 0.0
    
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    adj_weighted = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float64)
    adj_unweighted = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float64, weight=None)
    
    adj_weighted.sort_indices()
    adj_unweighted.sort_indices()
    
    degrees = np.array(adj_unweighted.sum(axis=1)).flatten().astype(np.float64)
    
    two_m = 2.0 * m
    n_comms = len(comm_list)
    
    comm_index_lists = [
        np.array([node_to_idx[node] for node in comm if node in node_to_idx], dtype=np.int64)
        for comm in comm_list
    ]
    
    # comm_masks[c, v] = True if node v is in community c
    comm_masks = np.zeros((n_comms, n), dtype=np.bool_)
    for c_idx, comm_indices in enumerate(comm_index_lists):
        if len(comm_indices) > 0:
            comm_masks[c_idx, comm_indices] = True
    
    # node_in_comm[u, c] = True if node u is in community c
    node_in_comm = comm_masks.T.copy()  # Shape: (n_nodes, n_comms)
    
    if HAS_NUMBA:
        alpha_matrix = _compute_alpha_matrix_numba(
            adj_unweighted.indptr, adj_unweighted.indices,
            comm_masks, node_in_comm, n, n_comms
        )
        
        Q = 0.0
        for c_idx, comm_indices in enumerate(comm_index_lists):
            if len(comm_indices) == 0:
                continue
            
            alpha = alpha_matrix[comm_indices, c_idx]
            
            Q += _compute_Q_for_community_numba(
                adj_weighted.indptr, adj_weighted.indices, adj_weighted.data,
                degrees, comm_indices, alpha, two_m
            )
        
        return Q / two_m
    else:
        return _compute_modularity_numpy(
            adj_weighted, adj_unweighted, degrees,
            comm_index_lists, comm_masks, node_in_comm, two_m, n, n_comms
        )


def _compute_modularity_numpy(
    adj_weighted: csr_matrix,
    adj_unweighted: csr_matrix,
    degrees: np.ndarray,
    comm_index_lists: List[np.ndarray],
    comm_masks: np.ndarray,
    node_in_comm: np.ndarray,
    two_m: float,
    n: int,
    n_comms: int
) -> float:
    """Fallback NumPy implementation."""
    
    # Compute d_{u,C} for all nodes and communities
    d_node_comm = adj_unweighted.dot(comm_masks.T.astype(np.float64))
    
    # Normalize alpha
    row_sums = d_node_comm.sum(axis=1, keepdims=True)
    alpha_matrix = np.divide(
        d_node_comm, row_sums,
        out=np.zeros_like(d_node_comm),
        where=row_sums != 0
    )
    
    # Handle nodes with no edges to community members
    zero_sum_mask = (row_sums.flatten() == 0)
    if np.any(zero_sum_mask):
        memberships = node_in_comm[zero_sum_mask].sum(axis=1, keepdims=True)
        uniform_alpha = np.divide(
            node_in_comm[zero_sum_mask].astype(np.float64),
            memberships,
            out=np.zeros_like(node_in_comm[zero_sum_mask], dtype=np.float64),
            where=memberships != 0
        )
        alpha_matrix[zero_sum_mask] = uniform_alpha
    
    Q = 0.0
    
    for c_idx, comm_indices in enumerate(comm_index_lists):
        if len(comm_indices) == 0:
            continue
        
        sub_adj = adj_weighted[comm_indices][:, comm_indices]
        A_sub = sub_adj.toarray()
        
        d_sub = degrees[comm_indices]
        null_model = np.outer(d_sub, d_sub) / two_m
        
        B = A_sub - null_model
        
        alpha_sub = alpha_matrix[comm_indices, c_idx]
        f_alpha = (alpha_sub[:, np.newaxis] + alpha_sub[np.newaxis, :]) / 2.0
        
        Q += (B * f_alpha).sum()
    
    return Q / two_m


def calculate_overlapping_modularity(
    G: nx.Graph,
    communities: Communities,
    method: str = 'auto'
) -> float:
    """
    Calculate overlapping modularity Q.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph
    communities : list of sets or NodeClustering
        The community partition
    method : str
        'auto' - uses Numba if available, else NumPy
        
    Returns
    -------
    float
        The overlapping modularity value Q in [-1, 1]
    """
    return overlapping_modularity_q0(G, communities)


def calculate_alpha(G: nx.Graph, node: int, community: Set, all_communities: List[Set]) -> float:
    """Calculate alpha value for a node with respect to a community."""
    d_uCj = sum(1 for v in community if G.has_edge(node, v))
    denominator = sum(
        sum(1 for v in comm if G.has_edge(node, v)) 
        for comm in all_communities
    )
    return d_uCj / denominator if denominator > 0 else 0.0
