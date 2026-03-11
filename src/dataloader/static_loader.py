"""
Static graph loader for community detection benchmarking.

Loads a static graph and wraps it as a TemporalGraph with steps=[] (single
snapshot).  This is equivalent to ``load_txt_dataset`` with
``initial_fraction=1.0`` and no temporal batches — the same edge-reading and
graph-building helpers are shared between both loaders.

The returned TemporalGraph has ``len()==1``, so ``iter_snapshots()`` yields
exactly once, making it fully compatible with the existing temporal pipeline
(``run_algorithm``, ``evaluate``, ``log_results``).
"""
import os
from typing import Optional

import networkx as nx
from cdlib import NodeClustering

from src.dataloader.data_reader import read_edges, build_graph
from src.factory.factory import TemporalGraph


# Built-in NetworkX graphs with known ground truth
BUILTIN_GRAPHS = {
    "karate": {
        "loader": nx.karate_club_graph,
        "ground_truth_attr": "club",
    },
    "football": {
        "loader": lambda: nx.read_gml(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "football.gml")
        ),
        "ground_truth_attr": "value",
    },
}


def _extract_ground_truth(G: nx.Graph, attr: str) -> Optional[NodeClustering]:
    """Extract ground truth NodeClustering from node attributes.

    Args:
        G: NetworkX graph with node attributes.
        attr: Node attribute name containing community labels.

    Returns:
        NodeClustering if the attribute exists on at least one node, else None.
    """
    membership = nx.get_node_attributes(G, attr)
    if not membership:
        return None

    # Group nodes by community label
    communities: dict = {}
    for node, label in membership.items():
        communities.setdefault(label, []).append(node)

    community_list = list(communities.values())
    return NodeClustering(community_list, G, "ground_truth")


def load_static_as_temporal(
    file_path: str,
    source_idx: int = 0,
    target_idx: int = 1,
    delimiter: str = " ",
    preload_fraction: float = 1.0,
    ground_truth_attr: Optional[str] = None,
) -> TemporalGraph:
    """Load a static graph file and return it as a 1-snapshot TemporalGraph.

    Uses the same ``read_edges`` / ``build_graph`` helpers as the temporal
    ``load_txt_dataset`` — the only difference is that *all* edges go into the
    base graph and ``steps`` is empty.

    Args:
        file_path: Path to edge list / CSV / TSV / GML file.
        source_idx: Column index for source node (default: 0).
        target_idx: Column index for target node (default: 1).
        delimiter: Field delimiter (default: space).
        preload_fraction: Fraction of edges to load (default: 1.0 = all).
            If < 1.0, only the first ``preload_fraction * total_edges``
            edges are used.
        ground_truth_attr: Node attribute for ground truth communities
            (if available).

    Returns:
        ``TemporalGraph(base_graph=G, steps=[], ...)``
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".gml":
        G = nx.read_gml(file_path)
        G = nx.convert_node_labels_to_integers(G)
    else:
        # Same edge reader as load_txt_dataset
        data, full_nodes = read_edges(file_path, source_idx, target_idx, delimiter)

        # Apply preload fraction (partial graph)
        if preload_fraction < 1.0:
            n_edges = max(1, int(len(data) * preload_fraction))
            data = data[:n_edges]

        # Same graph builder as load_txt_dataset (all edges → base graph)
        G = build_graph(data, full_nodes)

    # Extract ground truth if requested
    gt_clusterings = None
    if ground_truth_attr:
        gt = _extract_ground_truth(G, ground_truth_attr)
        if gt is not None:
            gt_clusterings = [gt]  # Single-element list for 1 snapshot

    return TemporalGraph(
        base_graph=G,
        steps=[],
        _ground_truth_clusterings=gt_clusterings,
    )


def load_builtin_graph(name: str) -> TemporalGraph:
    """Load a built-in NetworkX graph as a 1-snapshot TemporalGraph.

    Supported: ``'karate'`` (Zachary's karate club with ground truth).

    Args:
        name: Name of the built-in graph.

    Returns:
        ``TemporalGraph(base_graph=G, steps=[], ...)`` with ground truth
        if available.
    """
    if name not in BUILTIN_GRAPHS:
        available = ", ".join(sorted(BUILTIN_GRAPHS.keys()))
        raise ValueError(f"Unknown built-in graph '{name}'. Available: {available}")

    entry = BUILTIN_GRAPHS[name]
    G = entry["loader"]()

    # Convert to integer labels if needed
    if not all(isinstance(n, int) for n in G.nodes()):
        G = nx.convert_node_labels_to_integers(G)

    gt_clusterings = None
    gt_attr = entry.get("ground_truth_attr")
    if gt_attr:
        gt = _extract_ground_truth(G, gt_attr)
        if gt is not None:
            gt_clusterings = [gt]

    return TemporalGraph(
        base_graph=G,
        steps=[],
        _ground_truth_clusterings=gt_clusterings,
    )
