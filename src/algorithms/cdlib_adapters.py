"""
CDlib algorithm adapters.

Registers external CDlib algorithms in the project registry so they can be
referenced by name in YAML run configs, just like custom algorithms.

Each adapter is a thin ``register(...)`` call that points to the CDlib
function and declares its metadata. No wrapper classes needed — the factory
wraps them in ``StaticMethodWrapper`` automatically.
"""
from cdlib import algorithms

from src.algorithms.registry import register

# ── Overlapping ──────────────────────────────────────────────────────────

register(
    name="angel",
    algo_type="static",
    clustering_type="overlapping",
    default_params={"threshold": 0.25},
    description="ANGEL: A New Graph-based Entity Linking algorithm",
)(algorithms.angel)

register(
    name="demon",
    algo_type="static",
    clustering_type="overlapping",
    default_params={"epsilon": 0.25},
    description="DEMON: a local-first discovery method for overlapping communities",
)(algorithms.demon)

register(
    name="umstmo",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Universal Multi-Scale Community Detection",
)(algorithms.umstmo)

register(
    name="slpa",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Speaker-Listener Label Propagation Algorithm",
)(algorithms.slpa)

register(
    name="core_expansion",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Core Expansion algorithm for overlapping community detection",
)(algorithms.core_expansion)

register(
    name="graph_entropy",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Graph Entropy based community detection",
)(algorithms.graph_entropy)

register(
    name="coach",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="COACH: Clustering of Overlapping Communities using Hub structures",
)(algorithms.coach)

register(
    name="percomvc",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Permanence based Overlapping Community Detection",
)(algorithms.percomvc)

register(
    name="dpclus",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Density-Periphery based Clustering",
)(algorithms.dpclus)

register(
    name="ipca",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Iterative Principal Component Analysis",
)(algorithms.ipca)

register(
    name="dcs",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Distributed Community Search",
)(algorithms.dcs)

register(
    name="lfm",
    algo_type="static",
    clustering_type="overlapping",
    default_params={"alpha": 1.0},
    description="Lancichinetti-Fortunato-Radicchi method for overlapping communities",
)(algorithms.lfm)

register(
    name="ebgc",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Entropy-based Graph Clustering",
)(algorithms.ebgc)

# lais2 and walkscan live in cdlib.algorithms.overlapping_partition
from cdlib.algorithms import overlapping_partition  # noqa: E402

register(
    name="lais2",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Label Propagation with Improved Seed Selection",
)(overlapping_partition.lais2)

register(
    name="walkscan",
    algo_type="static",
    clustering_type="overlapping",
    default_params={},
    description="Walk-based SCAN algorithm for overlapping communities",
)(overlapping_partition.walkscan)
