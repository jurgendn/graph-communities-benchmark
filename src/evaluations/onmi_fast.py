"""Fast overlapping NMI compatible with CDlib's MGH interface."""

from typing import Iterable, Sequence

import numpy as np
from cdlib import NodeClustering
from cdlib.evaluation import MatchingResult
from scipy import sparse


def _as_cover(communities: NodeClustering | Sequence[Iterable]) -> list[set]:
    """Normalize a partition to a list of node sets."""
    if isinstance(communities, NodeClustering):
        raw_communities = communities.communities
    else:
        raw_communities = communities
    return [set(community) for community in raw_communities]


def _partial_entropy(prob: np.ndarray) -> np.ndarray:
    """Compute ``-p * log2(p)`` elementwise with zero handling."""
    result = np.zeros_like(prob, dtype=np.float64)
    mask = prob > 0
    if np.any(mask):
        result[mask] = -prob[mask] * np.log2(prob[mask])
    return result


def _binary_entropy(prob: np.ndarray) -> np.ndarray:
    """Compute binary entropy for probabilities in ``[0, 1]``."""
    return _partial_entropy(prob) + _partial_entropy(1.0 - prob)


def _build_membership_matrix(
    cover: list[set],
    node_to_index: dict,
    n_nodes: int,
) -> sparse.csr_matrix:
    """Build a sparse node-by-community membership matrix."""
    rows: list[int] = []
    cols: list[int] = []

    for community_index, community in enumerate(cover):
        rows.extend(node_to_index[node] for node in community)
        cols.extend([community_index] * len(community))

    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, len(cover)),
        dtype=np.int8,
    )


def _pairwise_conditional_entropies(
    intersections: np.ndarray,
    sizes_x: np.ndarray,
    sizes_y: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute conditional entropies for all cover pairs."""
    total = float(n_nodes)

    n11 = intersections.astype(np.float64)
    n10 = sizes_x[:, np.newaxis].astype(np.float64) - n11
    n01 = sizes_y[np.newaxis, :].astype(np.float64) - n11
    n00 = total - n11 - n10 - n01
    n00 = np.maximum(n00, 0.0)

    a = n00 / total
    b = n01 / total
    c = n10 / total
    d = n11 / total

    entropy_x = _binary_entropy(sizes_x.astype(np.float64) / total)
    entropy_y = _binary_entropy(sizes_y.astype(np.float64) / total)

    joint_entropy = (
        _partial_entropy(a)
        + _partial_entropy(b)
        + _partial_entropy(c)
        + _partial_entropy(d)
    )

    agree_entropy = _partial_entropy(a) + _partial_entropy(d)
    disagree_entropy = _partial_entropy(b) + _partial_entropy(c)

    use_reference = agree_entropy > disagree_entropy
    conditional = np.where(
        use_reference,
        joint_entropy - entropy_y[np.newaxis, :],
        entropy_x[:, np.newaxis],
    )

    return conditional, entropy_x, entropy_y


def _cover_entropy(entropy_per_community: np.ndarray) -> float:
    return float(np.sum(entropy_per_community))


def _cover_conditional_entropy(
    conditional_matrix: np.ndarray,
    entropy_per_community: np.ndarray,
    normalized: bool,
) -> float:
    if conditional_matrix.size == 0:
        return 0.0

    best_matches = np.min(conditional_matrix, axis=1)
    if not normalized:
        return float(np.sum(best_matches))

    normalized_best = np.ones_like(best_matches)
    nonzero = entropy_per_community > 0
    normalized_best[nonzero] = best_matches[nonzero] / entropy_per_community[nonzero]
    return float(np.mean(normalized_best))


def onmi_score(
    communities1: NodeClustering | Sequence[Iterable],
    communities2: NodeClustering | Sequence[Iterable],
    variant: str = "MGH",
) -> float:
    """Compute ONMI using formulas compatible with CDlib's internal implementation."""
    cover = _as_cover(communities1)
    reference_cover = _as_cover(communities2)

    if (not cover and reference_cover) or (cover and not reference_cover):
        return 0.0
    if cover == reference_cover:
        return 1.0

    all_nodes = set().union(*cover, *reference_cover) if (cover or reference_cover) else set()
    n_nodes = len(all_nodes)
    if n_nodes == 0:
        return 1.0

    if variant not in {"LFK", "MGH", "MGH_LFK"}:
        raise ValueError(f"Unknown ONMI variant: {variant}")

    node_to_index = {node: index for index, node in enumerate(all_nodes)}

    matrix_x = _build_membership_matrix(cover, node_to_index, n_nodes)
    matrix_y = _build_membership_matrix(reference_cover, node_to_index, n_nodes)

    intersections = (matrix_x.T @ matrix_y).toarray().astype(np.int32)
    sizes_x = np.asarray(matrix_x.sum(axis=0)).ravel().astype(np.int32)
    sizes_y = np.asarray(matrix_y.sum(axis=0)).ravel().astype(np.int32)

    conditional_xy, entropy_x, entropy_y = _pairwise_conditional_entropies(
        intersections,
        sizes_x,
        sizes_y,
        n_nodes,
    )
    conditional_yx, _, _ = _pairwise_conditional_entropies(
        intersections.T,
        sizes_y,
        sizes_x,
        n_nodes,
    )

    normalized = variant == "LFK"
    h_xy = _cover_conditional_entropy(conditional_xy, entropy_x, normalized)
    h_yx = _cover_conditional_entropy(conditional_yx, entropy_y, normalized)
    h_x = _cover_entropy(entropy_x)
    h_y = _cover_entropy(entropy_y)

    if variant == "LFK":
        score = 1.0 - 0.5 * (h_xy + h_yx)
    elif variant == "MGH_LFK":
        if h_x == 0 or h_y == 0:
            score = 1.0
        else:
            score = 1.0 - 0.5 * (h_xy / h_x + h_yx / h_y)
    else:
        denominator = max(h_x, h_y)
        if denominator == 0:
            score = 1.0
        else:
            mutual_information = 0.5 * (h_x - h_xy + h_y - h_yx)
            score = mutual_information / denominator

    if np.isnan(score):
        raise ValueError("ONMI computation produced NaN")

    if score < 0 and score > -1e-10:
        score = 0.0
    if score > 1 and score < 1 + 1e-10:
        score = 1.0

    if round(float(score), 2) < 0 or round(float(score), 2) > 1:
        raise ValueError(
            f"Invalid ONMI score {score} produced from HXY={h_xy}, HYX={h_yx}, HX={h_x}, HY={h_y}"
        )

    return float(score)


def overlapping_normalized_mutual_information_MGH_fast(
    first_partition: NodeClustering,
    second_partition: NodeClustering,
    normalization: str = "max",
) -> MatchingResult:
    """Fast replacement for CDlib's overlapping NMI wrapper."""
    if normalization == "max":
        variant = "MGH"
    elif normalization == "LFK":
        variant = "MGH_LFK"
    else:
        raise ValueError(
            "Wrong 'normalization' value. Please specify one among [max, LFK]."
        )

    return MatchingResult(score=onmi_score(first_partition, second_partition, variant=variant))
