"""
Modularity metric dispatcher for crisp and overlapping community detection.

Crisp partitions use the Girvan-Newman modularity (Newman & Girvan, 2004),
implemented in NetworkX.  Overlapping partitions use the cdlib overlap
modularity (the score from ``evaluation.modularity_overlap``) and the
custom Q0 defined in ``target_modularity.py``.
"""

from typing import Tuple

import networkx as nx
from cdlib import NodeClustering, evaluation

from src.evaluations.onmi_fast import (
    overlapping_normalized_mutual_information_MGH_fast,
)
from src.evaluations.target_modularity import overlapping_modularity_q0

CLUSTERING_TYPES = ("crisp", "overlapping")


def compute_modularity(
    graph: nx.Graph,
    clustering: NodeClustering,
    clustering_type: str,
) -> Tuple[float, float]:
    """
    Compute modularity metrics appropriate for the given clustering type.

    For **crisp** partitions:
    - cdlib_mod = Girvan-Newman modularity (networkx community.modularity)
    - q0_mod = 0 (not applicable for crisp partitions)

    For **overlapping** partitions:
    - cdlib_mod = cdlib modularity_overlap
    - q0_mod = custom Q0 from overlapping_modularity_q0

    Args:
        graph: The NetworkX graph on which clustering was performed.
        clustering: A ``NodeClustering`` object (cdlib).
        clustering_type: Either ``"crisp"`` or ``"overlapping"``.

    Returns:
        ``(cdlib_modularity, q0_modularity)`` — floats in approximately ``[-1, 1]``.

    Raises:
        ValueError: If ``clustering_type`` is not one of the accepted values.
    """
    if clustering_type not in CLUSTERING_TYPES:
        raise ValueError(
            f"Unknown clustering_type '{clustering_type}'. "
            f"Must be one of {CLUSTERING_TYPES}."
        )

    if clustering_type == "crisp":
        # Crisp: Girvan-Newman modularity, q0 = 0
        communities_as_sets = [set(c) for c in clustering.communities]
        try:
            gn_mod = nx.algorithms.community.modularity(graph, communities_as_sets)
        except Exception:
            gn_mod = 0.0
        return gn_mod, 0.0

    # Overlapping: cdlib modularity_overlap + custom Q0
    try:
        cdlib_mod = evaluation.modularity_overlap(graph, clustering).score
    except Exception:
        cdlib_mod = 0.0

    try:
        q0_mod = overlapping_modularity_q0(graph, clustering)
    except Exception:
        q0_mod = 0.0

    return cdlib_mod, q0_mod


def compute_nmi_from_ground_truth(
    graph: nx.Graph,
    clustering: NodeClustering,
    ground_truth_attr: str | None = None,
    gt_clustering: NodeClustering | None = None,
) -> float:
    """
    Compute NMI between detected communities and ground truth.

    For **crisp** ground truth (each node belongs to exactly one community):
    - Uses standard NMI (normalized_mutual_information)

    For **overlapping** ground truth (nodes can belong to multiple communities):
    - Uses ONMI-MGH (overlapping NMI by McDaid-Greene-Hurley)

    Ground truth format:
    - Crisp: graph.nodes[node]['label'] = 1 (int)
    - Overlapping: graph.nodes[node]['label'] = '1,23,4,5' (comma-separated string)

    Args:
        graph: NetworkX graph
        clustering: Detected communities as NodeClustering
        ground_truth_attr: Node attribute containing ground truth (optional if gt_clustering provided)
        gt_clustering: Pre-computed ground truth NodeClustering (takes precedence if provided)

    Returns:
        NMI score (0.0 if computation fails)
    """
    # Use pre-computed ground truth if available
    if gt_clustering is not None:
        try:
            # Determine if overlapping
            is_overlapping = any(len(comm) > 1 for comm in gt_clustering.communities)

            if is_overlapping:
                # Overlapping NMI
                result = overlapping_normalized_mutual_information_MGH_fast(
                    clustering, gt_clustering
                )
                return result.score if result.score is not None else 0.0
            else:
                # Crisp NMI
                result = evaluation.normalized_mutual_information(
                    clustering, gt_clustering
                )
                return result.score if result.score is not None else 0.0
        except Exception:
            return 0.0

    # Fall back to extracting from node attributes
    if ground_truth_attr is None:
        return 0.0
    # Parse ground truth from node attributes
    ground_truth = {}
    
    for node in graph.nodes():
        gt_val = graph.nodes[node].get(ground_truth_attr)
        if gt_val is None:
            continue
        
        # Handle different types
        if isinstance(gt_val, (int, float)):
            # Crisp: single community as int
            ground_truth[node] = {int(gt_val)}
        elif isinstance(gt_val, str):
            # Overlapping: comma-separated string
            try:
                communities = set(int(x.strip()) for x in gt_val.split(',') if x.strip())
                ground_truth[node] = communities
            except ValueError:
                continue
        elif isinstance(gt_val, (set, frozenset, list)):
            # Already a collection
            ground_truth[node] = set(gt_val)
    
    if not ground_truth:
        return 0.0
    
    # Determine if overlapping
    is_overlapping = any(len(comms) > 1 for comms in ground_truth.values())
    
    # Build NodeClustering from ground truth
    comm_nodes = {}
    for node, comms in ground_truth.items():
        for c in comms:
            comm_nodes.setdefault(c, []).append(node)
    
    if not comm_nodes:
        return 0.0
    
    gt_clustering = NodeClustering(
        list(comm_nodes.values()), 
        graph, 
        "ground_truth"
    )
    
    # Compute NMI based on clustering type
    try:
        if is_overlapping:
            # Overlapping NMI
            result = overlapping_normalized_mutual_information_MGH_fast(
                clustering, gt_clustering
            )
            return result.score if result.score is not None else 0.0
        else:
            # Crisp NMI
            result = evaluation.normalized_mutual_information(
                clustering, gt_clustering
            )
            return result.score if result.score is not None else 0.0
    except Exception:
        return 0.0
