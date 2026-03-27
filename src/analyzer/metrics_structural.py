"""Per-node structural metrics for overlap quality analysis.

All functions are pure: (graph, communities) -> per-node metric dicts.

References:
    Participation coefficient: Guimera & Amaral (2005), J. Stat. Mech. P02001
    Embeddedness: Lancichinetti, Fortunato & Kertesz (2009), New J. Phys. 11, 033015
    Betweenness centrality: Freeman (1977), Sociometry 40(1), 35-41
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx


def _node_community_sets(
    communities: List[List[Any]],
) -> Dict[Any, set[int]]:
    """Map each node to the set of community indices it belongs to."""
    mapping: Dict[Any, set[int]] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            mapping.setdefault(node, set()).add(idx)
    return mapping


def _community_edge_counts(
    node: Any,
    graph: nx.Graph,
    node_comm_sets: Dict[Any, set[int]],
) -> Dict[int, int]:
    """Count edges from *node* to each community."""
    counts: Dict[int, int] = {}
    for neighbor in graph.neighbors(node):
        for c in node_comm_sets.get(neighbor, set()):
            counts[c] = counts.get(c, 0) + 1
    return counts


def participation_coefficient(
    graph: nx.Graph,
    communities: List[List[Any]],
) -> Dict[Any, float]:
    """Participation coefficient for every node in *graph*.

    p_i = 1 - sum_c (k_ic / k_i)^2

    where k_ic = edges from node i to community c, k_i = degree of i.

    Returns:
        Dict mapping node -> participation coefficient.
    """
    node_comm_sets = _node_community_sets(communities)

    result: Dict[Any, float] = {}
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree == 0:
            result[node] = 0.0
            continue

        counts = _community_edge_counts(node, graph, node_comm_sets)
        sum_sq = sum((c / degree) ** 2 for c in counts.values())
        result[node] = 1.0 - sum_sq

    return result


def max_embeddedness(
    graph: nx.Graph,
    communities: List[List[Any]],
) -> Dict[Any, float]:
    """Maximum embeddedness for every node.

    e_ic = k_ic / k_i  (fraction of edges within community c).
    Returns max_c e_ic for each node.

    Overlapping nodes should have *low* max-embeddedness (not deeply
    interior to any single community).

    Returns:
        Dict mapping node -> max embeddedness across all communities.
    """
    node_comm_sets = _node_community_sets(communities)

    result: Dict[Any, float] = {}
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree == 0:
            result[node] = 0.0
            continue

        counts = _community_edge_counts(node, graph, node_comm_sets)
        if not counts:
            result[node] = 0.0
        else:
            result[node] = max(c / degree for c in counts.values())

    return result


def betweenness_centrality(
    graph: nx.Graph,
    normalized: bool = True,
    k: int | None = None,
) -> Dict[Any, float]:
    """Betweenness centrality for every node.

    Thin wrapper around ``networkx.betweenness_centrality``.
    For large graphs, set *k* to approximate with sampling.

    Returns:
        Dict mapping node -> betweenness centrality.
    """
    return nx.betweenness_centrality(graph, normalized=normalized, k=k)
