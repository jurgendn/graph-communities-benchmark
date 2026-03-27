"""Accuracy metrics for overlap quality analysis.

Omega index for ground-truth comparison and ONMI for temporal stability.

References:
    Omega index: Collins & Dent (1988), Multivar. Behav. Res. 23(2), 231-242
    ONMI: McDaid, Greene & Hurley (2011), via src.evaluation.onmi_fast
"""

from __future__ import annotations

from typing import Any, List

import networkx as nx
from cdlib import NodeClustering, evaluation

from src.evaluation.onmi_fast import onmi_score


def omega_index(
    detected: List[List[Any]],
    ground_truth: List[List[Any]],
    graph: nx.Graph,
) -> float:
    """Compute the Omega index between detected and ground-truth communities.

    Args:
        detected: Detected community assignments (list of node lists).
        ground_truth: Ground-truth community assignments.
        graph: The graph (needed for cdlib NodeClustering construction).

    Returns:
        Omega index score in [-1, 1].
    """
    nc_detected = NodeClustering(detected, graph, "detected")
    nc_gt = NodeClustering(ground_truth, graph, "ground_truth")

    try:
        result = evaluation.omega(nc_detected, nc_gt)
        return float(result.score) if result.score is not None else 0.0
    except Exception:
        return 0.0


def onmi_consecutive(
    communities_prev: List[List[Any]],
    communities_curr: List[List[Any]],
) -> float:
    """Compute ONMI between two consecutive snapshot community assignments.

    Measures temporal stability: how much the community structure changed.
    A higher value means more stable communities.

    Uses the fast ONMI implementation directly (no NodeClustering needed).

    Args:
        communities_prev: Communities at snapshot t-1.
        communities_curr: Communities at snapshot t.

    Returns:
        ONMI score in [0, 1].
    """
    try:
        return onmi_score(communities_prev, communities_curr, variant="MGH")
    except Exception:
        return 0.0
