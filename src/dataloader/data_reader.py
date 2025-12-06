import networkx as nx
import numpy as np

from src.factory.factory import TemporalChanges, TemporalGraph


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
        Tuple[nx.Graph, List[Dict[str, Any]]]:
            - Initial NetworkX graph
            - List of temporal changes

    Processing steps:
        1. Read data from file, skipping comment lines (starting with //)
        2. Split data into initial and remaining parts
        3. Create the initial graph from the initial data
        4. Generate batches of changes from the remaining data
        5. Each batch includes:
            - New edge insertions
            - Random edge deletions according to delete_insert_ratio

    Example:
        >>> graph, changes = load_txt_dataset(
        ...     file_path="data.txt",
        ...     source_idx=0,
        ...     target_idx=1,
        ...     batch_range=0.1,
        ...     initial_fraction=0.2,
        ...     max_steps=10,
        ...     load_full_nodes=True
        ... )
    """
    data = []
    full_nodes = set()

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    node1, node2 = parts[source_idx], parts[target_idx]
                    data.append((node1, node2))
                    full_nodes.update([node1, node2])

    split_point = int(len(data) * initial_fraction)
    initial_edges = data[:split_point]
    remaining_edges = data[split_point:]

    G = nx.Graph()
    if load_full_nodes:
        G.add_nodes_from(full_nodes)

    for node1, node2 in initial_edges:
        if G.has_edge(node1, node2):
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)

    initial_G = G.copy()

    batch_size = max(1, int(batch_range * len(data)))
    temporal_changes = []

    for batch_idx in range(min(num_batches, len(remaining_edges) // batch_size + 1)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_edges))

        if start_idx >= len(remaining_edges):
            break

        batch = remaining_edges[start_idx:end_idx]
        insertions = [(node1, node2, {"weight": 1}) for node1, node2 in batch]

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

        for node1, node2, data in insertions:
            weight = data["weight"]
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