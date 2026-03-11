import glob
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from cdlib import NodeClustering

from src.factory.factory import TemporalChanges, TemporalGraph


# ---------------------------------------------------------------------------
# Shared helpers — used by both dynamic (load_txt_dataset) and static loaders
# ---------------------------------------------------------------------------

def read_edges(
    file_path: str,
    source_idx: int = 0,
    target_idx: int = 1,
    delimiter: str = " ",
) -> Tuple[List[Tuple[str, str]], set]:
    """Read edge pairs from a text / CSV / TSV file.

    Skips blank lines and comment lines (starting with ``//``).

    Args:
        file_path: Path to the data file.
        source_idx: Column index for the source node.
        target_idx: Column index for the target node.
        delimiter: Field delimiter.

    Returns:
        A tuple of (edges, full_nodes) where *edges* is an ordered list of
        ``(source, target)`` string pairs and *full_nodes* is the set of all
        node identifiers encountered.
    """
    data: List[Tuple[str, str]] = []
    full_nodes: set = set()

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    node1, node2 = parts[source_idx], parts[target_idx]
                    data.append((node1, node2))
                    full_nodes.update([node1, node2])

    return data, full_nodes


def build_graph(
    edges: List[Tuple[str, str]],
    full_nodes: set | None = None,
) -> nx.Graph:
    """Build an undirected NetworkX graph from an edge list.

    Duplicate edges are accumulated as integer weights.

    Args:
        edges: Ordered list of ``(source, target)`` string pairs.
        full_nodes: If provided, pre-populate the graph with these nodes
            (useful when you want isolated nodes to exist).

    Returns:
        A ``nx.Graph`` with ``weight`` attributes on every edge.
    """
    G = nx.Graph()
    if full_nodes:
        G.add_nodes_from(full_nodes)

    for node1, node2 in edges:
        if G.has_edge(node1, node2):
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)

    return G


# ---------------------------------------------------------------------------
# Dynamic (temporal) loader
# ---------------------------------------------------------------------------

def load_txt_dataset(
    file_path: str,
    source_idx: int,
    target_idx: int,
    batch_range: float,
    initial_fraction: float,
    max_steps: int,
    load_full_nodes: bool,
    delimiter: str = " ",
    num_batches: int = 100,
    delete_insert_ratio: float = 0.8,
) -> TemporalGraph:
    """
    Load and process a dataset from a text file to create a dynamic graph and temporal changes.

    This function reads data from a text file, creates the initial graph, and generates
    batches of temporal changes (edge insertions/deletions) to simulate dynamic graph evolution.

    Args:
        file_path (str): Path to the data file
        source_idx (int): Index of the source node column in the file
        target_idx (int): Index of the target node column in the file
        batch_range (float): Fraction of data for each batch (0.0-1.0)
        initial_fraction (float): Fraction of data used to create the initial graph
        max_steps (int): Maximum number of temporal change steps
        load_full_nodes (bool): Whether to load all nodes into the initial graph
        num_batches (int): Maximum number of batches to create
        delete_insert_ratio (float): Ratio of edge deletions to insertions in each batch

    Returns:
        TemporalGraph with base graph and temporal change steps.

    Processing steps:
        1. Read data from file, skipping comment lines (starting with //)
        2. Split data into initial and remaining parts
        3. Create the initial graph from the initial data
        4. Generate batches of changes from the remaining data
        5. Each batch includes:
            - New edge insertions
            - Random edge deletions according to delete_insert_ratio

    Example:
        >>> tg = load_txt_dataset(
        ...     file_path="data.txt",
        ...     source_idx=0,
        ...     target_idx=1,
        ...     batch_range=0.1,
        ...     initial_fraction=0.2,
        ...     max_steps=10,
        ...     load_full_nodes=True
        ... )
    """
    data, full_nodes = read_edges(file_path, source_idx, target_idx, delimiter)

    split_point = int(len(data) * initial_fraction)
    initial_edges = data[:split_point]
    remaining_edges = data[split_point:]

    G = build_graph(initial_edges, full_nodes if load_full_nodes else None)

    initial_G = G.copy()

    batch_size = max(1, int(batch_range * len(data)))
    temporal_changes = []
    
    # Track current position in remaining_edges
    current_idx = 0
    
    while current_idx < len(remaining_edges) and len(temporal_changes) < num_batches:
        # Calculate end index, ensuring at least 1 edge is taken
        end_idx = min(current_idx + batch_size, len(remaining_edges))
        
        # Ensure we take at least 1 edge
        if end_idx <= current_idx:
            end_idx = current_idx + 1
        
        batch = remaining_edges[current_idx:end_idx]
        insertions = [(node1, node2, {"weight": 1}) for node1, node2 in batch]
        
        # Move current_idx forward to mark edges as processed
        current_idx = end_idx

        deletions = []
        current_edges = list(G.edges())
        if current_edges:
            num_deletions = min(
                int(len(insertions) * delete_insert_ratio), len(current_edges)
            )
            if num_deletions > 0:
                deletion_indices = np.random.choice(
                    len(current_edges), size=num_deletions, replace=False
                )
                deletions = [current_edges[i] for i in deletion_indices]

                G.remove_edges_from(deletions)

        for node1, node2, edge_data in insertions:
            weight = edge_data["weight"]
            if G.has_edge(node1, node2):
                G[node1][node2]["weight"] += weight
            else:
                G.add_edge(node1, node2, weight=weight)

        temporal_changes.append(
            TemporalChanges(
                deletions=deletions,
                insertions=insertions,
            )
        )

    if max_steps is not None:
        temporal_changes = temporal_changes[:max_steps]

    return TemporalGraph(base_graph=initial_G, steps=temporal_changes)


def node_membership_to_communities(raw_node_membership: Dict[int, str]) -> List[List[int]]:
    """Convert node membership dict to communities list.

    Args:
        raw_node_membership: Dict mapping node IDs to comma-separated community IDs

    Returns:
        List of communities, where each community is a list of node IDs
    """
    communities = defaultdict(list)
    for node_id, comms in raw_node_membership.items():
        for c in comms.split(","):
            communities[int(c)].append(int(node_id))
    max_c = max(communities.keys(), default=-1)
    return [communities[i] for i in range(max_c + 1)]


def load_lfr_folder(
    folder_path: str,
    ground_truth_attr: str = "communities",
    max_steps: int | None = None,
) -> TemporalGraph:
    """
    Load LFR benchmark graphs from a folder of GML files.

    Expects files named snapshot_t0.gml, snapshot_t1.gml, etc.
    Pre-computes ground truth NodeClustering objects for each snapshot.

    Args:
        folder_path: Path to folder containing GML snapshot files
        ground_truth_attr: Node attribute for ground truth (default: 'communities')
        max_steps: Maximum number of snapshots to load (default: all)

    Returns:
        TemporalGraph with pre-computed ground truth clusterings

    Example:
        >>> tg = load_lfr_folder(
        ...     folder_path="./data/synthetic_n_5000_1",
        ...     ground_truth_attr="communities",
        ...     max_steps=10
        ... )
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all GML files and extract indices
    gml_files = glob.glob(os.path.join(folder_path, "*.gml"))

    if not gml_files:
        raise ValueError(f"No .gml files found in {folder_path}")

    # Extract snapshot indices and sort
    snapshot_files = []
    for filepath in gml_files:
        filename = os.path.basename(filepath)
        # Expected format: snapshot_t{index}.gml
        if filename.startswith("snapshot_t") and filename.endswith(".gml"):
            try:
                index_str = filename[len("snapshot_t"):-len(".gml")]
                index = int(index_str)
                snapshot_files.append((index, filepath))
            except ValueError:
                continue

    if not snapshot_files:
        raise ValueError(
            "No valid snapshot files found. Expected format: snapshot_t0.gml, "
            "snapshot_t1.gml, etc."
        )

    # Sort by index
    snapshot_files.sort(key=lambda x: x[0])

    # Load all snapshots and pre-compute ground truth
    snapshots = []
    ground_truth_clusterings = []
    for index, filepath in snapshot_files:
        G = nx.read_gml(filepath)
        G = nx.convert_node_labels_to_integers(G)
        snapshots.append(G)

        # Pre-compute ground truth NodeClustering
        raw_node_membership = nx.get_node_attributes(G, ground_truth_attr)
        communities = node_membership_to_communities(raw_node_membership)
        ground_truth_clustering = NodeClustering(communities, G, "ground_truth")
        ground_truth_clusterings.append(ground_truth_clustering)

    # Limit to max_steps if specified
    if max_steps is not None:
        snapshots = snapshots[:max_steps + 1]  # +1 because first snapshot is base
        ground_truth_clusterings = ground_truth_clusterings[:max_steps + 1]

    if len(snapshots) < 2:
        raise ValueError("Need at least 2 snapshots for temporal graph")

    # Detect temporal changes between consecutive snapshots
    steps = []
    for i in range(1, len(snapshots)):
        prev_graph = snapshots[i - 1]
        curr_graph = snapshots[i]

        # Edge changes
        prev_edges = set(prev_graph.edges())
        curr_edges = set(curr_graph.edges())

        insertions = list(curr_edges - prev_edges)
        deletions = list(prev_edges - curr_edges)

        steps.append(TemporalChanges(
            deletions=deletions,
            insertions=insertions,
        ))

    # Create TemporalGraph with pre-computed ground truth
    base_graph = snapshots[0]
    tg = TemporalGraph(base_graph=base_graph, steps=steps, _ground_truth_clusterings=ground_truth_clusterings)

    return tg