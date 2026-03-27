"""Typed models for clustering artifact payloads and analysis reports."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Artifact payload models — lightweight JSON metadata for searchability.
# The actual NodeClustering objects (with graph + communities) live in
# the companion ``clusterings.pkl`` pickle file.
# ---------------------------------------------------------------------------


class SnapshotClustering(BaseModel):
    """Per-snapshot summary stored in JSON metadata (no communities/graph)."""

    snapshot_index: int
    num_communities: int
    overlap: bool = False


class ClusteringArtifactPayload(BaseModel):
    """Lightweight metadata stored as ``clustering_payload.json``.

    Keeps provenance metadata so downstream tools can filter, group,
    and reproduce results without unpickling the full clusterings.
    """

    # --- provenance ---
    algorithm: str
    algorithm_type: str  # "static" | "dynamic"
    clustering_type: str  # "crisp" | "overlapping"
    algorithm_params: Dict[str, Any] = {}
    dataset_config: Dict[str, Any] = {}
    dataset: str
    benchmark_mode: str  # "dynamic" | "static"
    num_snapshots: int

    # --- per-snapshot summaries ---
    snapshots: List[SnapshotClustering]

    # --- optional metric summaries (useful for quick filtering) ---
    avg_runtime: float | None = None
    avg_modularity: float | None = None
    avg_nmi: float | None = None


# ---------------------------------------------------------------------------
# Overlap quality report models (Phase 2)
# ---------------------------------------------------------------------------


class MetricComparison(BaseModel):
    """Mann-Whitney U comparison between overlapping / non-overlapping nodes."""

    metric_name: str
    group_a_label: str = "overlap"
    group_b_label: str = "non_overlap"
    group_a_n: int = 0
    group_b_n: int = 0
    group_a_median: float | None = None
    group_b_median: float | None = None
    group_a_mean: float | None = None
    group_b_mean: float | None = None
    u_statistic: float | None = None
    p_value: float | None = None
    rank_biserial: float | None = None
    error: str | None = None


class SnapshotOverlapReport(BaseModel):
    """Per-snapshot structural analysis of overlap quality."""

    snapshot_index: int
    num_nodes: int
    num_edges: int
    num_communities: int
    num_overlapping_nodes: int
    overlap_fraction: float

    # Structural metric comparisons (overlap vs non-overlap)
    participation_coefficient: MetricComparison | None = None
    max_embeddedness: MetricComparison | None = None
    betweenness_centrality: MetricComparison | None = None

    # Ground-truth accuracy (only for LFR synthetics)
    omega_index: float | None = None


class TemporalStabilityEntry(BaseModel):
    """ONMI between consecutive snapshots t-1 -> t."""

    snapshot_from: int
    snapshot_to: int
    onmi: float


class OverlapQualityReport(BaseModel):
    """Top-level report from the overlap-quality analyzer."""

    algorithm: str
    dataset: str
    benchmark_mode: str
    clustering_type: str
    num_snapshots: int

    # Per-snapshot structural reports
    snapshot_reports: List[SnapshotOverlapReport] = []

    # Temporal stability (for multi-snapshot runs)
    temporal_stability: List[TemporalStabilityEntry] = []

    # Membership distribution summary (always available)
    community_counts: List[int] = []
    overlap_node_counts: List[int] = []
