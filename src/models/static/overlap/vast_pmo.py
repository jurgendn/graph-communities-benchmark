"""VastPMO: Overlapping community detection using Parameterized Modularity Overlap."""

import time
from typing import Dict, List

import networkx as nx
import numpy as np
from cdlib import NodeClustering
from scipy.sparse import csr_matrix

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.communities import (
    IntermediateResults,
    MethodDynamicResults,
    OverlapPartitionModel,
)
from src.factory.factory import TemporalGraph


class PMO:
    """Algorithm 4: Structural Parameterized Modularity Overlap."""
 
    def __init__(self, theta: float):
        self.theta = theta

    def run(self, graph: nx.Graph, current_backbone: Dict[int, set]):
        """
        Parameters
        ----------
        graph    : Undirected (possibly weighted) graph.
        current_backbone : Crisp partition from Louvain — {community_id: set of nodes}.
 
        Returns
        -------
        (node_to_communities, community_to_nodes)
        """
        start_time = time.time()
        nodes = list(graph.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        adj = csr_matrix(
            nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr"),
            dtype=np.float64,
        )
        deg = np.asarray(adj.sum(axis=1)).ravel()
        D = float(deg.sum())  # d = 2m
 
        # Initialise memberships from backbone
        node_comms = [set() for _ in nodes]
        comm = {}
        for cid, members in current_backbone.items():
            comm[cid] = {idx[v] for v in members if v in idx}
            for i in comm[cid]:
                node_comms[i].add(cid)
 
        # Main loop (lines 2–11)
        changed = True
        while changed:
            changed = False
            vol = {c: float(deg[list(m)].sum()) for c, m in comm.items() if m}
            ref1 = {c: vol[c] / len(m) for c, m in comm.items() if m}
 
            for u in range(len(nodes)):
                if deg[u] == 0:
                    continue
                row = adj.getrow(u)
                nbrs, wts = row.indices, row.data
 
                candidates = set()
                for v in nbrs:
                    candidates.update(node_comms[v])
                candidates -= node_comms[u]
 
                for c in candidates:
                    m = comm.get(c)
                    if not m:
                        continue
                    e = sum(w for v, w in zip(nbrs, wts) if v in m)
                    thr = max(ref1[c], self.theta * deg[u] * vol[c] / D) if D else 0.0
                    if e >= thr:
                        node_comms[u].add(c)
                        m.add(u)
                        changed = True
        executed_time = time.time() - start_time
        
        # --- Build output ---
        # Build both views
        result_c2n = {c: {nodes[i] for i in m} for c, m in comm.items() if m}
        result_n2c = {}
        for i, comms in enumerate(node_comms):
            if comms:
                result_n2c[nodes[i]] = set(comms)

        pmo_result = OverlapPartitionModel(
            node_to_communities=result_n2c, community_to_nodes=result_c2n
        )

        results = MethodDynamicResults()
        results.update_intermediate_results(
            IntermediateResults(
                runtime=executed_time,
                modularity=0.0,
                affected_nodes=graph.number_of_nodes(),
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

    def __init__(self, louvain_resolution: float = 1.0, theta: float = 0.85):
        """
        Initialize VastPMO parameters.

        Args:
            louvain_resolution: Resolution parameter for the initial Louvain step
            theta: Modularity threshold parameter
        """
        self.louvain_resolution = louvain_resolution
        self.pmo = PMO(theta=theta)

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
        community_to_nodes = {
            cid: set(members) for cid, members in enumerate(louvain_communities)
        }
        overlap_result, _ = self.pmo.run(graph=G, current_backbone=community_to_nodes)
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
) -> NodeClustering:
    """
    Convenience function for VastPMO community detection on a single graph.

    Args:
        G: NetworkX graph
        louvain_resolution: Resolution parameter for the initial Louvain step
        theta: Modularity threshold parameter

    Returns:
        NodeClustering object with detected overlapping communities
    """
    model = VastPMO(louvain_resolution=louvain_resolution, theta=theta)
    return model._process_snapshot(G)
