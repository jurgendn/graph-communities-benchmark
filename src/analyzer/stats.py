"""Statistical comparison utilities for overlap quality analysis."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats


def mann_whitney_compare(
    group_a: np.ndarray,
    group_b: np.ndarray,
    labels: tuple[str, str] = ("overlap", "non_overlap"),
) -> Dict[str, Any]:
    """Compare two groups using Mann-Whitney U test with effect size.

    Args:
        group_a: Metric values for first group (e.g. overlapping nodes).
        group_b: Metric values for second group (e.g. non-overlapping nodes).
        labels: Names for the two groups.

    Returns:
        Dict with U statistic, p-value, rank-biserial correlation,
        and descriptive statistics for both groups.
    """
    result: Dict[str, Any] = {
        "group_a_label": labels[0],
        "group_b_label": labels[1],
        "group_a_n": len(group_a),
        "group_b_n": len(group_b),
    }

    if len(group_a) < 1 or len(group_b) < 1:
        result["error"] = "Insufficient data for comparison"
        return result

    # Descriptive stats
    for key, arr in [("group_a", group_a), ("group_b", group_b)]:
        result[f"{key}_median"] = float(np.median(arr))
        result[f"{key}_mean"] = float(np.mean(arr))
        result[f"{key}_q25"] = float(np.percentile(arr, 25))
        result[f"{key}_q75"] = float(np.percentile(arr, 75))

    # Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(
            group_a, group_b, alternative="two-sided"
        )
        result["u_statistic"] = float(u_stat)
        result["p_value"] = float(p_value)

        # Rank-biserial correlation as effect size
        # r_rb = 1 - (2U)/(n1*n2); range [-1, 1]
        n1, n2 = len(group_a), len(group_b)
        result["rank_biserial"] = 1.0 - (2.0 * u_stat) / (n1 * n2)
    except ValueError as e:
        result["error"] = str(e)

    return result
