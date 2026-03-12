"""VastPMO: Overlapping community detection using Parameterized Modularity Overlap."""

import os
import time
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from cdlib import NodeClustering
from scipy.sparse import csr_matrix

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.communities import (
    IntermediateResults,
    MethodDynamicResults,
    NonOverlapPartitionModel,
    OverlapPartitionModel,
)
from src.factory.factory import TemporalGraph


class PMO:
    """
    High-performance PMO algorithm using only NumPy/SciPy (no Numba).
    Good for environments where Numba can't be installed.
    """

    __slots__ = ("theta", "lambda_param", "epsilon", "r", "n_jobs")

    def __init__(
        self,
        theta: float,
        lambda_param: float = 0.1,
        epsilon: float = 1e-5,
        r: int = 5,
        n_jobs: int = -1,
    ):
        self.theta = theta
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.r = r
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)

    def _build_sparse(self, graph: nx.Graph):
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = np.array(nodes, dtype=object)

        adj = nx.to_scipy_sparse_array(
            graph, nodelist=nodes, weight="weight", format="csr"
        )
        # NetworkX may return scipy.sparse.csr_array, which lacks getrow().
        # Convert to csr_matrix so row access works across SciPy versions.
        adj = csr_matrix(adj.astype(np.float64))
        degrees = np.array(adj.sum(axis=1)).flatten()

        return adj, node_to_idx, idx_to_node, degrees

    def _compute_stats_vectorized(
        self, adj: csr_matrix, comm_members: dict, degrees: np.ndarray
    ):
        stats = {}
        for cid, members in comm_members.items():
            if len(members) == 0:
                stats[cid] = (members, 0, 0.0, 0.0)
                continue

            size = len(members)
            volume = degrees[members].sum()

            # Submatrix extraction for internal edges
            sub = adj[members][:, members]
            internal_sum = sub.sum()
            ref_1 = internal_sum / size if size > 0 else 0.0

            stats[cid] = (set(members), size, volume, ref_1)

        return stats

    def run(
        self,
        graph: nx.Graph,
        current_backbone: NonOverlapPartitionModel,
        previous_overlap: Optional[OverlapPartitionModel] = None,
        previous_backbone: Optional[NonOverlapPartitionModel] = None,
        delta_nodes: Optional[set] = None,
    ) -> Tuple[OverlapPartitionModel, MethodDynamicResults]:
        start_time = time.time()
        adj, node_to_idx, idx_to_node, degrees = self._build_sparse(graph)
        n_nodes = len(idx_to_node)
        total_degree = degrees.sum()

        epsilon = self.epsilon
        lambda_param = self.lambda_param
        r_limit = self.r

        node_comms = [set() for _ in range(n_nodes)]
        comm_members = {}
        prev_node_comms = None

        if previous_overlap is None:
            frontier_idx = np.arange(n_nodes, dtype=np.int32)
            for cid, members in current_backbone.community_to_nodes.items():
                idxs = np.array(
                    [node_to_idx[n] for n in members if n in node_to_idx],
                    dtype=np.int32,
                )
                comm_members[cid] = idxs
                for idx in idxs:
                    node_comms[idx].add(cid)
        else:
            frontier_set = set()
            if previous_backbone:
                for node in graph.nodes():
                    if node not in node_to_idx:
                        continue
                    curr_c = next(
                        iter(current_backbone.node_to_communities.get(node, {-1})), -1
                    )
                    prev_c = next(
                        iter(previous_backbone.node_to_communities.get(node, {-1})), -1
                    )
                    if curr_c != prev_c:
                        frontier_set.add(node_to_idx[node])
            else:
                frontier_set = set(range(n_nodes))

            if delta_nodes:
                frontier_set.update(
                    node_to_idx[n] for n in delta_nodes if n in node_to_idx
                )

            frontier_idx = np.array(list(frontier_set), dtype=np.int32)

            for cid, members in previous_overlap.community_to_nodes.items():
                idxs = np.array(
                    [node_to_idx[n] for n in members if n in node_to_idx],
                    dtype=np.int32,
                )
                comm_members[cid] = idxs
                for idx in idxs:
                    node_comms[idx].add(cid)

            prev_node_comms = [set(s) for s in node_comms]

            for u_idx in frontier_idx:
                node = idx_to_node[u_idx]
                primary_comms = current_backbone.node_to_communities.get(node)
                if primary_comms:
                    pc = next(iter(primary_comms), None)
                    if pc is not None:
                        node_comms[u_idx].add(pc)
                        if pc not in comm_members:
                            comm_members[pc] = np.array([u_idx], dtype=np.int32)
                        elif u_idx not in set(comm_members[pc]):
                            comm_members[pc] = np.append(comm_members[pc], u_idx)

        frontier_idx = frontier_idx[degrees[frontier_idx] > 0]

        while True:
            comm_stats = self._compute_stats_vectorized(adj, comm_members, degrees)
            updates = []

            for u_idx in frontier_idx:
                current_comms = node_comms[u_idx]
                if len(current_comms) >= r_limit:
                    continue

                d_u = degrees[u_idx]
                u_row = adj.getrow(u_idx)
                neighbors = u_row.indices
                weights = u_row.data

                adjacent_comms = set()
                for v in neighbors:
                    adjacent_comms.update(node_comms[v])

                candidates = adjacent_comms - current_comms
                prev_comms = prev_node_comms[u_idx] if prev_node_comms else set()

                for cid in candidates:
                    member_set, size, vol, ref_1 = comm_stats.get(
                        cid, (set(), 0, 0.0, 0.0)
                    )
                    if size == 0:
                        continue

                    d_C_u = sum(
                        w for v, w in zip(neighbors, weights) if v in member_set
                    )
                    ref_2 = (vol * d_u) / total_degree if total_degree > 0 else 0.0

                    delta_q = d_C_u - max(ref_1, ref_2)
                    bonus = lambda_param if cid in prev_comms else 0.0

                    if delta_q + bonus > epsilon:
                        updates.append((u_idx, cid))

            if not updates:
                break

            for u_idx, cid in updates:
                if len(node_comms[u_idx]) < r_limit:
                    node_comms[u_idx].add(cid)
                    if cid in comm_members:
                        if u_idx not in set(comm_members[cid]):
                            comm_members[cid] = np.append(comm_members[cid], u_idx)
                    else:
                        comm_members[cid] = np.array([u_idx], dtype=np.int32)

        result_n2c = {idx_to_node[i]: set(c) for i, c in enumerate(node_comms) if c}
        result_c2n = {
            cid: {idx_to_node[i] for i in m}
            for cid, m in comm_members.items()
            if len(m) > 0
        }

        pmo_result = OverlapPartitionModel(
            node_to_communities=result_n2c, community_to_nodes=result_c2n
        )

        results = MethodDynamicResults()
        results.update_intermediate_results(
            IntermediateResults(
                runtime=time.time() - start_time,
                modularity=0.0,
                affected_nodes=len(frontier_idx),
                num_communities=len(result_c2n),
            )
        )

        return pmo_result, results


class VastPMO(CommunityDetectionAlgorithm):
    """
    Detect overlapping communities using Parameterized Modularity Overlap (PMO).

    The algorithm:
    1. Detects initial non-overlapping communities using the Louvain method
    2. Refines them into overlapping communities via parameterized modularity
       optimization, allowing nodes to join additional communities when the
       modularity gain exceeds a threshold
    """

    def __init__(
        self,
        louvain_resolution: float = 1.0,
        theta: float = 0.85,
        lambda_param: float = 0.1,
        epsilon: float = 1e-5,
        r: int = 5,
        n_jobs: int = -1,
    ):
        """
        Initialize VastPMO parameters.

        Args:
            theta: Modularity threshold parameter
            lambda_param: Stability bonus for previously assigned communities (default 0.1)
            epsilon: Convergence threshold for modularity gain (default 1e-5)
            r: Maximum number of communities a node can belong to (default 5)
            n_jobs: Number of parallel jobs; -1 uses all available CPUs (default -1)
        """
        self.louvain_resolution = louvain_resolution
        self.pmo = PMO(
            theta=theta,
            lambda_param=lambda_param,
            epsilon=epsilon,
            r=r,
            n_jobs=n_jobs,
        )

    def _process_snapshot(self, G: nx.Graph) -> NodeClustering:
        """
        Detect overlapping communities in a single graph snapshot.

        Args:
            G: NetworkX graph

        Returns:
            NodeClustering object with detected overlapping communities
        """
        louvain_communities = nx.algorithms.community.louvain_communities(
            G=G, resolution=self.louvain_resolution
        )
        node_to_communities = {
            node: {cid}
            for cid, members in enumerate(louvain_communities)
            for node in members
        }
        community_to_nodes = {
            cid: set(members) for cid, members in enumerate(louvain_communities)
        }
        backbone = NonOverlapPartitionModel(
            node_to_communities=node_to_communities,
            community_to_nodes=community_to_nodes,
        )
        overlap_result, _ = self.pmo.run(graph=G, current_backbone=backbone)
        return overlap_result.to_cdlib_node_clustering()

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """Run VastPMO on each snapshot of the temporal graph."""
        results = []
        for snapshot in tg.iter_snapshots():
            results.append(self._process_snapshot(snapshot))
        return results


def vast_pmo(
    G: nx.Graph,
    louvain_resolution: float = 1.0,
    theta: float = 0.85,
    lambda_param: float = 0.1,
    epsilon: float = 1e-5,
    r: int = 5,
    n_jobs: int = -1,
) -> NodeClustering:
    """
    Convenience function for VastPMO community detection on a single graph.

    Args:
        G: NetworkX graph
        theta: Modularity threshold parameter
        lambda_param: Stability bonus for previously assigned communities (default 0.1)
        epsilon: Convergence threshold for modularity gain (default 1e-5)
        r: Maximum number of communities a node can belong to (default 5)
        n_jobs: Number of parallel jobs; -1 uses all available CPUs (default -1)

    Returns:
        NodeClustering object with detected overlapping communities
    """
    model = VastPMO(
        louvain_resolution=louvain_resolution,
        theta=theta,
        lambda_param=lambda_param,
        epsilon=epsilon,
        r=r,
        n_jobs=n_jobs,
    )
    return model._process_snapshot(G)
