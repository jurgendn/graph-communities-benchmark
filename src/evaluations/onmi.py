"""Compatibility layer for overlapping NMI utilities.

This module now delegates to the fast implementation in
``src.evaluations.onmi_fast`` so existing imports keep working while the
evaluation pipeline uses the optimized code path.
"""

from typing import Iterable, Sequence

from cdlib import NodeClustering
from cdlib.evaluation import MatchingResult

from src.evaluations.onmi_fast import (
    onmi_score,
    overlapping_normalized_mutual_information_MGH_fast,
)


def onmi_lfk(
    communities1: NodeClustering | Sequence[Iterable],
    communities2: NodeClustering | Sequence[Iterable],
) -> float:
    return onmi_score(communities1, communities2, variant="LFK")


def onmi_mcdaid(
    communities1: NodeClustering | Sequence[Iterable],
    communities2: NodeClustering | Sequence[Iterable],
) -> float:
    return onmi_score(communities1, communities2, variant="MGH")


def onmi_max(
    communities1: NodeClustering | Sequence[Iterable],
    communities2: NodeClustering | Sequence[Iterable],
) -> float:
    return onmi_score(communities1, communities2, variant="MGH")


def compute_onmi(
    communities1: NodeClustering | Sequence[Iterable],
    communities2: NodeClustering | Sequence[Iterable],
    variant: str = "max",
) -> MatchingResult:
    normalized_variant = variant.lower()
    if normalized_variant in {"max", "mcdaid", "mgh"}:
        normalization = "max"
    elif normalized_variant in {"lfk", "mgh_lfk"}:
        normalization = "LFK"
    else:
        raise ValueError(f"Unknown ONMI variant: {variant}")
    return overlapping_normalized_mutual_information_MGH_fast(
        communities1,
        communities2,
        normalization=normalization,
    )


class ONMI:
    """Backward-compatible callable wrapper around the fast ONMI implementation."""

    VALID_VARIANTS = {"lfk", "mcdaid", "max", "mgh", "mgh_lfk"}

    def __init__(self, variant: str = "max"):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of {self.VALID_VARIANTS}")
        self.variant = variant

    def compute(
        self,
        communities1: NodeClustering | Sequence[Iterable],
        communities2: NodeClustering | Sequence[Iterable],
    ) -> float:
        if self.variant in {"lfk", "mgh_lfk"}:
            return onmi_lfk(communities1, communities2)
        if self.variant in {"mcdaid", "max", "mgh"}:
            return onmi_mcdaid(communities1, communities2)
        return onmi_max(communities1, communities2)

    def __call__(
        self,
        communities1: NodeClustering | Sequence[Iterable],
        communities2: NodeClustering | Sequence[Iterable],
    ) -> float:
        return self.compute(communities1, communities2)
