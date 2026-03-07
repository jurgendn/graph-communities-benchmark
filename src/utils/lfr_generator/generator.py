import bisect
import math
import random
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import networkx as nx
from cdlib import NodeClustering


def unweighted_undirected_lfr_graph(
    num_nodes: int,
    average_k: float,
    max_degree: int,
    mu: float,
    com_size_min: int,
    com_size_max: int,
    seed: Optional[int] = None,
    tau: float = 2.0,
    tau2: float = 1.0,
    overlapping_nodes: int = 0,
    overlap_membership: int = 0,
    fixed_range: bool = True,
    excess: bool = False,
    defect: bool = False,
    avg_clustering: float = 0.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, ...]]]:
    """
    Generate an unweighted, undirected LFR benchmark graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network (node IDs start at 0).
    average_k : float
        Average degree of the nodes.
    max_degree : int
        Maximum degree of any node.
    mu : float
        Mixing parameter (fraction of edges that are inter-community).
        Must be between 0 and 1.
    com_size_min : int
        Minimum community size.
    com_size_max :  int
        Maximum community size.
    seed : int, optional
        Random seed for reproducibility.
    tau : float, default=2.0
        Minus exponent for the degree distribution power law.
    tau2 : float, default=1.0
        Minus exponent for the community size distribution power law.
    overlapping_nodes : int, default=0
        Number of overlapping nodes (nodes belonging to multiple communities).
    overlap_membership : int, default=0
        Number of communities each overlapping node belongs to.
    fixed_range : bool, default=True
        If True, use com_size_min/max strictly; otherwise distribution determines range.
    excess :  bool, default=False
        If True, mixing parameter is a lower bound.
    defect :  bool, default=False
        If True, mixing parameter is an upper bound.
    avg_clustering : float, default=0.0
        Target average clustering coefficient.  If > 0, rewiring is performed.

    Returns
    -------
    edges : List[Tuple[int, int]]
        List of edges as (source, target) tuples.
    community_memberships : List[Tuple[int, ... ]]
        Tuple of community IDs for each node (index corresponds to node ID).
    """
    if seed is not None:
        random.seed(seed)

    # Validate parameters
    if mu < 0 or mu > 1:
        raise ValueError("Mixing parameter mu must be between 0 and 1")
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    if average_k <= 0 or max_degree <= 0:
        raise ValueError("Degrees must be positive")
    if overlapping_nodes > num_nodes:
        raise ValueError("Overlapping nodes cannot exceed total nodes")
    if excess and defect:
        raise ValueError("Cannot use both excess and defect options")

    # Generate degree sequence
    degree_seq = _generate_degree_sequence(num_nodes, average_k, max_degree, tau)

    # Generate community structure
    member_matrix, num_seq = _assign_community_membership(
        num_nodes,
        degree_seq,
        mu,
        com_size_min,
        com_size_max,
        tau2,
        overlapping_nodes,
        overlap_membership,
        fixed_range,
        excess,
        defect,
    )

    # Build the graph
    E = _build_graph(num_nodes, degree_seq, member_matrix, mu, excess, defect)

    # Apply clustering coefficient enhancement if requested
    if avg_clustering > 0:
        _enhance_clustering(E, member_matrix, avg_clustering)

    # Convert adjacency to edge list
    edges = []
    for u in range(len(E)):
        for v in E[u]:
            if u < v:
                edges.append((u, v))

    # Build community memberships for each node
    member_list = _build_member_list(num_nodes, member_matrix)
    community_memberships = [tuple(member_list[i]) for i in range(num_nodes)]

    return edges, community_memberships


def _generate_degree_sequence(
    num_nodes: int, average_k: float, max_degree: int, tau: float
) -> List[int]:
    """Generate a power-law degree sequence."""

    # Find minimum degree using bisection to match target average
    min_degree = _solve_dmin(max_degree, average_k, tau)

    # Generate cumulative distribution
    cumulative = _powerlaw_cumulative(max_degree, min_degree, tau)

    # Sample degrees
    degree_seq = []
    for _ in range(num_nodes):
        r = random.random()
        idx = bisect.bisect_left(cumulative, r)
        degree = idx + min_degree
        degree_seq.append(min(degree, max_degree))

    degree_seq.sort()

    # Ensure sum is even (required for simple graph)
    if sum(degree_seq) % 2 != 0:
        max_idx = degree_seq.index(max(degree_seq))
        degree_seq[max_idx] -= 1

    return degree_seq


def _solve_dmin(max_degree: int, average_k: float, tau: float) -> int:
    """Find minimum degree to achieve target average degree."""

    def avg_degree(dmax: int, dmin: float, gamma: float) -> float:
        if abs(gamma + 1) > 1e-10:
            def int_gamma(x: float) -> float:
                return pow(x, gamma + 1) / (gamma + 1)
            
            def int_gamma1(x: float) -> float:
                return pow(x, gamma + 2) / (gamma + 2)
        else:
            def int_gamma(x: float) -> float:
                return math.log(x)
            
            def int_gamma1(x: float) -> float:
                return x * math.log(x) - x

        norm = int_gamma(dmax) - int_gamma(dmin)
        if abs(norm) < 1e-10:
            return dmin
        return (int_gamma1(dmax) - int_gamma1(dmin)) / norm

    gamma = -tau
    dmin_l: float = 1
    dmin_r: float = max_degree

    for _ in range(100):  # Binary search iterations
        mid = (dmin_l + dmin_r) / 2
        avg = avg_degree(max_degree, mid, gamma)
        if avg > average_k:
            dmin_l = mid
        else:
            dmin_r = mid
        if abs(avg - average_k) < 1e-7:
            break

    return max(1, int(dmin_l))


def _powerlaw_cumulative(n: int, min_val: int, tau: float) -> List[float]:
    """Generate cumulative distribution for power law."""
    probs = []
    total = sum(pow(1.0 / h, tau) for h in range(min_val, n + 1))

    cumsum = 0.0
    for h in range(min_val, n + 1):
        cumsum += pow(1.0 / h, tau) / total
        probs.append(cumsum)

    return probs


def _assign_community_membership(
    num_nodes: int,
    degree_seq: List[int],
    mu: float,
    nmin: int,
    nmax: int,
    tau2: float,
    overlapping_nodes: int,
    overlap_membership: int,
    fixed_range: bool,
    excess: bool,
    defect: bool,
) -> Tuple[List[List[int]], List[int]]:
    """Assign nodes to communities."""

    # Calculate internal degrees
    internal_degree_seq = []
    max_internal = 0

    for degree in degree_seq:
        internal = (1 - mu) * degree
        int_internal = int(internal)
        if random.random() < (internal - int_internal):
            int_internal += 1

        if excess:
            while int_internal / degree < (1 - mu) and int_internal < degree:
                int_internal += 1
        elif defect:
            while int_internal / degree > (1 - mu) and int_internal > 0:
                int_internal -= 1

        internal_degree_seq.append(int_internal)
        max_internal = max(max_internal, int_internal)

    # Determine community size range
    if not fixed_range:
        nmax = max(max_internal + 1, nmax)
        nmin = max(nmin, 3)

    # Generate community sizes
    num_seq = _generate_community_sizes(
        num_nodes,
        nmin,
        nmax,
        tau2,
        overlapping_nodes,
        overlap_membership,
        max_internal,
        fixed_range,
    )

    # Build bipartite matching between nodes and communities
    member_matrix: List[List[int]] = _build_bipartite_membership(
        num_nodes, num_seq, overlapping_nodes, overlap_membership
    )

    return member_matrix, num_seq


def _generate_community_sizes(
    num_nodes: int,
    nmin: int,
    nmax: int,
    tau2: float,
    overlapping_nodes: int,
    overlap_membership: int,
    max_internal: int,
    fixed_range: bool,
) -> List[int]:
    """Generate community sizes following power law."""

    cumulative = _powerlaw_cumulative(nmax, nmin, tau2)
    num_seq = []
    total = 0
    target = num_nodes + overlapping_nodes * (max(overlap_membership, 1) - 1)

    # Possibly add one large community
    if not fixed_range and (max_internal + 1) > nmin:
        num_seq.append(max_internal + 1)
        total = max_internal + 1

    while total < target:
        r = random.random()
        idx = bisect.bisect_left(cumulative, r)
        size = idx + nmin

        if total + size <= target:
            num_seq.append(size)
            total += size
        else:
            break

    # Adjust smallest community to match exactly
    if total < target and num_seq:
        min_idx = num_seq.index(min(num_seq))
        num_seq[min_idx] += target - total

    if not num_seq:
        num_seq = [num_nodes]

    return num_seq


def _build_bipartite_membership(
    num_nodes: int, num_seq: List[int], overlapping_nodes: int, overlap_membership: int
) -> List[List[int]]:
    """Build community membership matrix."""

    member_matrix: List[List[int]] = [[] for _ in range(len(num_seq))]

    # Determine memberships for each node
    member_numbers = []
    for i in range(num_nodes):
        if i < overlapping_nodes and overlap_membership > 0:
            member_numbers.append(overlap_membership)
        else:
            member_numbers.append(1)

    # Simple greedy assignment
    node_list = list(range(num_nodes))
    random.shuffle(node_list)

    # Track community capacities
    capacity = num_seq.copy()

    for node in node_list:
        num_memberships = member_numbers[node]

        # Find communities with capacity
        available = [c for c in range(len(num_seq)) if capacity[c] > 0]

        if len(available) >= num_memberships:
            chosen = random.sample(available, num_memberships)
        else:
            chosen = available

        for c in chosen:
            member_matrix[c].append(node)
            capacity[c] -= 1

    # Sort members in each community
    for c in range(len(member_matrix)):
        member_matrix[c].sort()

    return member_matrix


def _build_member_list(
    num_nodes: int, member_matrix: List[List[int]]
) -> List[List[int]]:
    """Build list of communities for each node."""
    member_list: List[List[int]] = [[] for _ in range(num_nodes)]

    for c, members in enumerate(member_matrix):
        for node in members:
            member_list[node].append(c)

    return member_list


def _they_are_mate(a: int, b: int, member_list: List[List[int]]) -> bool:
    """Check if two nodes share a community."""
    for c in member_list[a]:
        if c in member_list[b]:
            return True
    return False


def _build_graph(
    num_nodes: int,
    degree_seq: List[int],
    member_matrix: List[List[int]],
    mu: float,
    excess: bool,
    defect: bool,
) -> List[Set[int]]:
    """Build the graph with community structure."""

    E: List[Set[int]] = [set() for _ in range(num_nodes)]
    member_list = _build_member_list(num_nodes, member_matrix)

    # Calculate internal degrees
    internal_degree_seq = []
    for _, degree in enumerate(degree_seq):
        internal = int((1 - mu) * degree + 0.5)
        internal_degree_seq.append(internal)

    # Build intra-community edges
    for c, members in enumerate(member_matrix):
        if len(members) < 2:
            continue

        # Calculate internal degrees for this community
        com_degrees = []
        for node in members:
            num_communities = len(member_list[node])
            internal = internal_degree_seq[node] // num_communities
            com_degrees.append(min(internal, len(members) - 1))

        # Ensure sum is even
        total = sum(com_degrees)
        if total % 2 != 0:
            max_idx = com_degrees.index(max(com_degrees))
            if com_degrees[max_idx] > 0:
                com_degrees[max_idx] -= 1

        # Build edges within community using configuration model
        _configuration_model_subgraph(E, members, com_degrees)

    # Build inter-community edges
    _add_external_edges(E, degree_seq, internal_degree_seq, member_list, mu)

    return E


def _configuration_model_subgraph(
    E: List[Set[int]], nodes: List[int], degrees: List[int]
) -> None:
    """Build a random graph on given nodes with given degrees."""

    if len(nodes) < 2:
        return

    # Create stubs
    stubs = []
    for i, node in enumerate(nodes):
        stubs.extend([node] * degrees[i])

    random.shuffle(stubs)

    # Match stubs
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v and v not in E[u]:
            E[u].add(v)
            E[v].add(u)

    # Rewire to remove self-loops and multi-edges
    _rewire_subgraph(E, nodes, 10)


def _rewire_subgraph(E: List[Set[int]], nodes: List[int], iterations: int) -> None:
    """Rewire edges to randomize while preserving degrees."""

    if len(nodes) < 4:
        return

    node_set = set(nodes)

    for _ in range(iterations):
        for u in nodes:
            neighbors = [v for v in E[u] if v in node_set]
            if len(neighbors) < 2:
                continue

            # Pick a neighbor to swap
            v = random.choice(neighbors)

            # Find another node in community
            candidates = [n for n in nodes if n != u and n != v and n not in E[u]]
            if not candidates:
                continue

            w = random.choice(candidates)

            # Find a neighbor of w that's not connected to v
            w_neighbors = [
                x for x in E[w] if x in node_set and x != u and x not in E[v]
            ]
            if not w_neighbors:
                continue

            x = random.choice(w_neighbors)

            # Perform the swap:  (u-v, w-x) -> (u-w, v-x)
            E[u].discard(v)
            E[v].discard(u)
            E[w].discard(x)
            E[x].discard(w)

            E[u].add(w)
            E[w].add(u)
            E[v].add(x)
            E[x].add(v)


def _add_external_edges(
    E: List[Set[int]],
    degree_seq: List[int],
    internal_degree_seq: List[int],
    member_list: List[List[int]],
    mu: float,
) -> None:
    """Add inter-community edges."""

    num_nodes = len(degree_seq)

    # Calculate external degree for each node
    external_degrees = []
    for i in range(num_nodes):
        # current = len(E[i])
        target = degree_seq[i]
        external = target - internal_degree_seq[i]
        external_degrees.append(max(0, external))

    # Create stubs for external edges
    stubs = []
    for i in range(num_nodes):
        stubs.extend([i] * external_degrees[i])

    random.shuffle(stubs)

    # Match stubs, preferring nodes from different communities
    attempts = 0
    max_attempts = len(stubs) * 10
    i = 0

    while i < len(stubs) - 1 and attempts < max_attempts:
        u = stubs[i]
        j = i + 1

        while j < len(stubs):
            v = stubs[j]

            if u != v and v not in E[u] and not _they_are_mate(u, v, member_list):
                E[u].add(v)
                E[v].add(u)
                # Remove matched stubs
                stubs.pop(j)
                stubs.pop(i)
                break
            j += 1
        else:
            i += 1

        attempts += 1

    # Handle remaining stubs (may need to add edges even to same community)
    while len(stubs) >= 2:
        u = stubs.pop()

        for idx in range(len(stubs) - 1, -1, -1):
            v = stubs[idx]
            if u != v and v not in E[u]:
                E[u].add(v)
                E[v].add(u)
                stubs.pop(idx)
                break


def _enhance_clustering(
    E: List[Set[int]], member_matrix: List[List[int]], target_cc: float
) -> None:
    """Enhance clustering coefficient through rewiring."""

    # member_list = _build_member_list(len(E), member_matrix)
    current_cc = _compute_clustering_coefficient(E)

    if current_cc >= target_cc:
        return

    max_iterations = 50

    for iteration in range(max_iterations):
        improved = False

        for _ in range(len(E)):
            # Random edge swap to increase triangles
            u = random.randint(0, len(E) - 1)
            if len(E[u]) < 2:
                continue

            neighbors = list(E[u])
            v = random.choice(neighbors)

            # Find potential new neighbor that would create more triangles
            candidates = []
            for w in range(len(E)):
                if w != u and w != v and w not in E[u]:
                    # Count potential triangles
                    common = len(E[u] & E[w])
                    if common > 0:
                        candidates.append((w, common))

            if not candidates:
                continue

            # Choose the one that adds most triangles
            candidates.sort(key=lambda x: -x[1])
            new_neighbor = candidates[0][0]

            # Find an edge from new_neighbor to swap
            for x in E[new_neighbor]:
                if x != u and x not in E[v] and x != v:
                    # Swap:  remove u-v and new_neighbor-x, add u-new_neighbor and v-x
                    old_cc = _compute_local_cc(E, [u, v, new_neighbor, x])

                    E[u].discard(v)
                    E[v].discard(u)
                    E[new_neighbor].discard(x)
                    E[x].discard(new_neighbor)

                    E[u].add(new_neighbor)
                    E[new_neighbor].add(u)
                    E[v].add(x)
                    E[x].add(v)

                    new_cc = _compute_local_cc(E, [u, v, new_neighbor, x])

                    if new_cc <= old_cc:
                        # Revert
                        E[u].discard(new_neighbor)
                        E[new_neighbor].discard(u)
                        E[v].discard(x)
                        E[x].discard(v)

                        E[u].add(v)
                        E[v].add(u)
                        E[new_neighbor].add(x)
                        E[x].add(new_neighbor)
                    else:
                        improved = True

                    break

        current_cc = _compute_clustering_coefficient(E)
        if current_cc >= target_cc or not improved:
            break


def _compute_clustering_coefficient(E: List[Set[int]]) -> float:
    """Compute global average clustering coefficient."""
    cc_sum = 0.0
    count = 0

    for u in range(len(E)):
        if len(E[u]) < 2:
            continue

        neighbors = list(E[u])
        triangles = 0

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in E[neighbors[i]]:
                    triangles += 1

        possible = len(neighbors) * (len(neighbors) - 1) / 2
        if possible > 0:
            cc_sum += triangles / possible
            count += 1

    return cc_sum / count if count > 0 else 0.0


def _compute_local_cc(E: List[Set[int]], nodes: List[int]) -> float:
    """Compute average clustering coefficient for specific nodes."""
    cc_sum = 0.0
    count = 0

    for u in nodes:
        if len(E[u]) < 2:
            continue

        neighbors = list(E[u])
        triangles = 0

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in E[neighbors[i]]:
                    triangles += 1

        possible = len(neighbors) * (len(neighbors) - 1) / 2
        if possible > 0:
            cc_sum += triangles / possible
            count += 1

    return cc_sum / count if count > 0 else 0.0


# Convenience function to convert to NetworkX graph
def to_networkx(edges, community_memberships):
    """
    Convert LFR graph to NetworkX format.

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        Edge list from unweighted_undirected_lfr_graph.
    community_memberships : List[Tuple[int, ... ]]
        Community memberships from unweighted_undirected_lfr_graph.

    Returns
    -------
    G : networkx.Graph
        NetworkX graph with 'communities' node attribute.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function")

    G = nx.Graph()

    # Add nodes with community attributes
    for node, communities in enumerate(community_memberships):
        G.add_node(node, communities=communities)

    # Add edges
    G.add_edges_from(edges)

    return G


# ==================== Temporal Graph Generation Functions ====================


def generate_temporal_graph_from_static(
    edges: List[Tuple[int, int]],
    num_timestamps: int,
    temporal_distribution: str = "uniform",
    seed: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Generate a temporal graph by assigning timestamps to static edges.

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        Static edge list.
    num_timestamps : int
        Number of distinct timestamps to use.
    temporal_distribution : str, default="uniform"
        How to distribute edges across time:
        - "uniform": Each edge randomly assigned to any timestamp
        - "sequential": Edges added in order across timestamps
        - "burst": Multiple edges appear in bursts at certain times
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    """
    if seed is not None:
        random.seed(seed)

    temporal_edges = []

    if temporal_distribution == "uniform":
        for u, v in edges:
            t = random.randint(0, num_timestamps - 1)
            temporal_edges.append((u, v, t))

    elif temporal_distribution == "sequential":
        edges_per_time = len(edges) // num_timestamps
        for i, (u, v) in enumerate(edges):
            t = min(i // max(edges_per_time, 1), num_timestamps - 1)
            temporal_edges.append((u, v, t))

    elif temporal_distribution == "burst":
        # Create bursts at 20% of timestamps with 80% of edges
        num_burst_times = max(1, num_timestamps // 5)
        burst_times = random.sample(range(num_timestamps), num_burst_times)
        num_burst_edges = int(len(edges) * 0.8)

        # Assign burst edges
        for i in range(num_burst_edges):
            u, v = edges[i]
            t = random.choice(burst_times)
            temporal_edges.append((u, v, t))

        # Assign remaining edges to non-burst times
        non_burst_times = [t for t in range(num_timestamps) if t not in burst_times]
        if non_burst_times:
            for i in range(num_burst_edges, len(edges)):
                u, v = edges[i]
                t = random.choice(non_burst_times)
                temporal_edges.append((u, v, t))
        else:
            for i in range(num_burst_edges, len(edges)):
                u, v = edges[i]
                t = random.choice(burst_times)
                temporal_edges.append((u, v, t))

    else:
        raise ValueError(f"Unknown temporal_distribution: {temporal_distribution}")

    # Sort by timestamp
    temporal_edges.sort(key=lambda x: x[2])

    return temporal_edges


def generate_temporal_lfr_graphs(
    num_nodes: int,
    num_timestamps: int,
    average_k: float,
    max_degree: int,
    mu: float,
    com_size_min: int,
    com_size_max: int,
    tau: float = 2.0,
    tau2: float = 1.0,
    overlapping_nodes: int = 0,
    overlap_membership: int = 0,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], List[List[Tuple[int, ...]]]]:
    """
    Generate temporal graphs where each snapshot is an independent LFR graph.
    
    Each timestamp gets a new LFR benchmark graph with the same number of nodes
    but potentially different community structure and edges.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes (constant across all timestamps).
    num_timestamps : int
        Number of temporal snapshots to generate.
    average_k : float
        Average degree of nodes in each snapshot.
    max_degree : int
        Maximum degree of any node in each snapshot.
    mu : float
        Mixing parameter (fraction of inter-community edges).
    com_size_min : int
        Minimum community size.
    com_size_max : int
        Maximum community size.
    tau : float, default=2.0
        Exponent for degree distribution power law.
    tau2 : float, default=1.0
        Exponent for community size distribution power law.
    overlapping_nodes : int, default=0
        Number of overlapping nodes per snapshot.
    overlap_membership : int, default=0
        Number of communities each overlapping node belongs to.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    community_history : List[List[Tuple[int, ...]]]
        Community memberships for each timestamp.
        community_history[t][i] is the tuple of community IDs for node i at timestamp t.
    """
    if seed is not None:
        random.seed(seed)
    
    temporal_edges = []
    community_history = []
    
    for t in range(num_timestamps):
        # Generate a new LFR graph for this timestamp
        # Use different seed for each timestamp if seed is provided
        snapshot_seed = None if seed is None else seed + t
        
        edges, communities = unweighted_undirected_lfr_graph(
            num_nodes=num_nodes,
            average_k=average_k,
            max_degree=max_degree,
            mu=mu,
            com_size_min=com_size_min,
            com_size_max=com_size_max,
            seed=snapshot_seed,
            tau=tau,
            tau2=tau2,
            overlapping_nodes=overlapping_nodes,
            overlap_membership=overlap_membership,
        )
        
        # Add timestamp to edges
        for u, v in edges:
            temporal_edges.append((u, v, t))
        
        # Store community memberships for this timestamp
        community_history.append(communities)
    
    # Sort by timestamp (already sorted but explicit)
    temporal_edges.sort(key=lambda x: x[2])
    
    return temporal_edges, community_history


def generate_labeled_temporal_graph(
    num_nodes: int,
    num_edges: int,
    num_timestamps: int,
    num_node_labels: int,
    num_edge_labels: int,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """
    Generate a random labeled temporal graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of temporal edges to generate.
    num_timestamps : int
        Number of distinct timestamps.
    num_node_labels : int
        Number of distinct node labels.
    num_edge_labels : int
        Number of distinct edge labels.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int, int]]
        List of temporal edges as (source, target, timestamp, edge_label) tuples.
    node_labels : List[int]
        List of node labels (index corresponds to node ID).
    """
    if seed is not None:
        random.seed(seed)

    # Assign node labels
    node_labels = [random.randint(0, num_node_labels - 1) for _ in range(num_nodes)]

    # Generate temporal edges with labels
    temporal_edges = []
    seen_edges = set()

    for _ in range(num_edges):
        attempts = 0
        while attempts < 100:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            t = random.randint(0, num_timestamps - 1)

            if u != v and (u, v, t) not in seen_edges and (v, u, t) not in seen_edges:
                edge_label = random.randint(0, num_edge_labels - 1)
                temporal_edges.append((u, v, t, edge_label))
                seen_edges.add((u, v, t))
                break

            attempts += 1

    # Sort by timestamp
    temporal_edges.sort(key=lambda x: x[2])

    return temporal_edges, node_labels


def generate_evolving_community_graph(
    num_nodes: int,
    num_communities: int,
    num_timestamps: int,
    intra_community_prob: float = 0.3,
    inter_community_prob: float = 0.05,
    community_evolution_rate: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
    """
    Generate a temporal graph with evolving community structure.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_communities : int
        Number of communities.
    num_timestamps : int
        Number of distinct timestamps.
    intra_community_prob : float, default=0.3
        Probability of edge within same community.
    inter_community_prob : float, default=0.05
        Probability of edge between different communities.
    community_evolution_rate : float, default=0.1
        Fraction of nodes that change community at each timestamp.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    community_history : List[List[int]]
        List of community assignments at each timestamp.
        community_history[t][i] is the community of node i at timestamp t.
    """
    if seed is not None:
        random.seed(seed)

    # Initial community assignment
    community_assignment = [i % num_communities for i in range(num_nodes)]
    random.shuffle(community_assignment)

    community_history = []
    temporal_edges = []

    for t in range(num_timestamps):
        # Store current community assignment
        community_history.append(community_assignment.copy())

        # Generate edges based on current community structure
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                same_community = community_assignment[u] == community_assignment[v]

                if same_community:
                    prob = intra_community_prob
                else:
                    prob = inter_community_prob

                if random.random() < prob:
                    temporal_edges.append((u, v, t))

        # Evolve communities for next timestamp
        if t < num_timestamps - 1:
            num_changes = int(num_nodes * community_evolution_rate)
            nodes_to_change = random.sample(range(num_nodes), num_changes)

            for node in nodes_to_change:
                new_community = random.randint(0, num_communities - 1)
                community_assignment[node] = new_community

    return temporal_edges, community_history


def generate_growing_network_temporal(
    initial_nodes: int,
    final_nodes: int,
    num_timestamps: int,
    edges_per_node: int = 2,
    attachment_type: str = "preferential",
    node_labels: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    Generate a growing network with preferential or random attachment.

    Parameters
    ----------
    initial_nodes : int
        Number of nodes in initial network.
    final_nodes : int
        Final number of nodes after growth.
    num_timestamps : int
        Number of distinct timestamps.
    edges_per_node : int, default=2
        Number of edges each new node makes.
    attachment_type : str, default="preferential"
        Type of attachment: "preferential" or "random".
    node_labels : List[int], optional
        Labels for nodes. If None, generated randomly.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    node_labels : List[int]
        List of node labels for all nodes.
    """
    if seed is not None:
        random.seed(seed)

    if final_nodes < initial_nodes:
        raise ValueError("final_nodes must be >= initial_nodes")

    # Initialize node labels
    if node_labels is None:
        num_label_types = max(3, final_nodes // 10)
        node_labels = [random.randint(0, num_label_types - 1) for _ in range(final_nodes)]
    elif len(node_labels) < final_nodes:
        raise ValueError("node_labels must have length >= final_nodes")

    temporal_edges = []
    degree = [0] * final_nodes

    # Create initial complete graph
    for u in range(initial_nodes):
        for v in range(u + 1, initial_nodes):
            temporal_edges.append((u, v, 0))
            degree[u] += 1
            degree[v] += 1

    # Add nodes over time
    nodes_to_add = final_nodes - initial_nodes
    if nodes_to_add > 0:
        timestamps_per_node = num_timestamps / max(nodes_to_add, 1)

        for i in range(nodes_to_add):
            new_node = initial_nodes + i
            t = int(i * timestamps_per_node) + 1

            if t >= num_timestamps:
                t = num_timestamps - 1

            # Select nodes to connect to
            existing_nodes = list(range(new_node))

            if attachment_type == "preferential":
                # Preferential attachment: probability proportional to degree
                total_degree = sum(degree[n] for n in existing_nodes)
                if total_degree > 0:
                    targets = []
                    for _ in range(min(edges_per_node, len(existing_nodes))):
                        r = random.random() * total_degree
                        cumsum = 0
                        for node in existing_nodes:
                            cumsum += degree[node]
                            if cumsum >= r and node not in targets:
                                targets.append(node)
                                break

                        if not targets or len(targets) < _:
                            # Fallback to random
                            remaining = [n for n in existing_nodes if n not in targets]
                            if remaining:
                                targets.append(random.choice(remaining))
                else:
                    targets = random.sample(existing_nodes, min(edges_per_node, len(existing_nodes)))

            else:  # random attachment
                targets = random.sample(existing_nodes, min(edges_per_node, len(existing_nodes)))

            # Add edges
            for target in targets:
                temporal_edges.append((target, new_node, t))
                degree[target] += 1
                degree[new_node] += 1

    return temporal_edges, node_labels


def generate_dynamic_labeled_network(
    num_nodes: int,
    num_timestamps: int,
    edge_prob: float = 0.1,
    num_node_labels: int = 5,
    num_edge_labels: int = 3,
    node_label_evolution_rate: float = 0.05,
    edge_label_temporal_pattern: str = "random",
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[List[int]]]:
    """
    Generate a dynamic labeled network with evolving node and edge labels.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_timestamps : int
        Number of distinct timestamps.
    edge_prob : float, default=0.1
        Base probability of edge existence at each timestamp.
    num_node_labels : int, default=5
        Number of distinct node label types.
    num_edge_labels : int, default=3
        Number of distinct edge label types.
    node_label_evolution_rate : float, default=0.05
        Fraction of nodes that change labels at each timestamp.
    edge_label_temporal_pattern : str, default="random"
        Pattern for edge labels: "random", "sequential", "correlated".
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int, int]]
        List of temporal edges as (source, target, timestamp, edge_label) tuples.
    node_label_history : List[List[int]]
        List of node labels at each timestamp.
        node_label_history[t][i] is the label of node i at timestamp t.
    """
    if seed is not None:
        random.seed(seed)

    # Initialize node labels
    node_labels = [random.randint(0, num_node_labels - 1) for _ in range(num_nodes)]
    node_label_history = []
    temporal_edges = []

    for t in range(num_timestamps):
        # Store current node labels
        node_label_history.append(node_labels.copy())

        # Generate edges for this timestamp
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if random.random() < edge_prob:
                    # Assign edge label based on pattern
                    if edge_label_temporal_pattern == "random":
                        edge_label = random.randint(0, num_edge_labels - 1)
                    elif edge_label_temporal_pattern == "sequential":
                        edge_label = t % num_edge_labels
                    elif edge_label_temporal_pattern == "correlated":
                        # Correlate with node labels
                        edge_label = (node_labels[u] + node_labels[v]) % num_edge_labels
                    else:
                        edge_label = 0

                    temporal_edges.append((u, v, t, edge_label))

        # Evolve node labels for next timestamp
        if t < num_timestamps - 1:
            num_changes = int(num_nodes * node_label_evolution_rate)
            nodes_to_change = random.sample(range(num_nodes), num_changes)

            for node in nodes_to_change:
                new_label = random.randint(0, num_node_labels - 1)
                node_labels[node] = new_label

    return temporal_edges, node_label_history


def save_temporal_graph_to_file(
    temporal_edges: List[Tuple[int, int, int]],
    filename: str,
    include_labels: bool = False,
) -> None:
    """
    Save temporal graph to file.

    Parameters
    ----------
    temporal_edges : List[Tuple[int, int, int]] or List[Tuple[int, int, int, int]]
        Temporal edges, optionally with edge labels.
    filename : str
        Output filename.
    include_labels : bool, default=False
        Whether edges include labels (4-tuples vs 3-tuples).
    """
    with open(filename, 'w') as f:
        if include_labels:
            f.write("# source target timestamp edge_label\n")
            for edge in temporal_edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")
        else:
            f.write("# source target timestamp\n")
            for edge in temporal_edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")


def load_temporal_graph_from_file(
    filename: str,
    has_labels: bool = False,
) -> List[Tuple]:
    """
    Load temporal graph from file.

    Parameters
    ----------
    filename : str
        Input filename.
    has_labels : bool, default=False
        Whether the file includes edge labels.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]] or List[Tuple[int, int, int, int]]
        Temporal edges, optionally with edge labels.
    """
    temporal_edges = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if has_labels and len(parts) >= 4:
                u, v, t, label = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                temporal_edges.append((u, v, t, label))
            elif not has_labels and len(parts) >= 3:
                u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
                temporal_edges.append((u, v, t))

    return temporal_edges


def save_temporal_graph_with_communities(
    temporal_edges: List[Tuple[int, int, int]],
    community_history: List[List[Tuple[int, ...]]],
    edges_filename: str,
    nodes_filename: str,
) -> None:
    """
    Save temporal graph with community labels as node attributes.

    Parameters
    ----------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    community_history : List[List[Tuple[int, ...]]]
        Community memberships for each timestamp.
        community_history[t][i] is the tuple of community IDs for node i at timestamp t.
    edges_filename : str
        Output filename for edges.
    nodes_filename : str
        Output filename for node attributes.
    """
    # Save edges
    with open(edges_filename, 'w') as f:
        f.write("# source target timestamp\n")
        for u, v, t in temporal_edges:
            f.write(f"{u} {v} {t}\n")
    
    # Save node communities for each timestamp
    with open(nodes_filename, 'w') as f:
        f.write("# timestamp node_id communities (comma-separated)\n")
        for t, communities in enumerate(community_history):
            for node_id, node_communities in enumerate(communities):
                # Convert tuple of communities to comma-separated string
                communities_str = ",".join(map(str, node_communities))
                f.write(f"{t} {node_id} {communities_str}\n")


def load_temporal_graph_with_communities(
    edges_filename: str,
    nodes_filename: str,
) -> Tuple[List[Tuple[int, int, int]], List[List[Tuple[int, ...]]]]:
    """
    Load temporal graph with community labels.

    Parameters
    ----------
    edges_filename : str
        Input filename for edges.
    nodes_filename : str
        Input filename for node attributes.

    Returns
    -------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    community_history : List[List[Tuple[int, ...]]]
        Community memberships for each timestamp.
    """
    # Load edges
    temporal_edges = load_temporal_graph_from_file(edges_filename, has_labels=False)
    
    # Load node communities
    community_dict = {}  # {timestamp: {node_id: tuple of communities}}
    
    with open(nodes_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                t = int(parts[0])
                node_id = int(parts[1])
                communities_str = parts[2]
                
                # Parse comma-separated communities
                communities = tuple(int(c) for c in communities_str.split(','))
                
                if t not in community_dict:
                    community_dict[t] = {}
                community_dict[t][node_id] = communities
    
    # Convert to list format
    community_history = []
    for t in sorted(community_dict.keys()):
        max_node = max(community_dict[t].keys())
        timestamp_communities = [() for _ in range(max_node + 1)]
        for node_id, communities in community_dict[t].items():
            timestamp_communities[node_id] = communities
        community_history.append(timestamp_communities)
    
    return temporal_edges, community_history


def temporal_to_networkx_snapshots(
    temporal_edges: List[Tuple[int, int, int]],
    community_history: Optional[List[List[Tuple[int, ...]]]] = None,
    num_timestamps: Optional[int] = None,
) -> List:
    """
    Convert temporal edges to NetworkX graph snapshots with community attributes.

    Parameters
    ----------
    temporal_edges : List[Tuple[int, int, int]]
        List of temporal edges as (source, target, timestamp) tuples.
    community_history : List[List[Tuple[int, ...]]], optional
        Community memberships for each timestamp.
        If provided, adds 'community' and 'communities' attributes to nodes.
    num_timestamps : int, optional
        Number of timestamps. If None, inferred from temporal_edges.

    Returns
    -------
    snapshots : List[networkx.Graph]
        List of NetworkX graphs, one for each timestamp.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function")
    
    # Determine number of timestamps
    if num_timestamps is None:
        if temporal_edges:
            num_timestamps = max(e[2] for e in temporal_edges) + 1
        else:
            num_timestamps = 0
    
    # Initialize snapshots
    snapshots = [nx.Graph() for _ in range(num_timestamps)]
    
    # Add edges to snapshots
    for u, v, t in temporal_edges:
        snapshots[t].add_edge(u, v)
    
    # Add community attributes if provided
    if community_history is not None:
        for t, communities in enumerate(community_history):
            if t < len(snapshots):
                for node_id, node_communities in enumerate(communities):
                    # Add node with community attributes
                    snapshots[t].add_node(
                        node_id,
                        communities=",".join(map(str, node_communities))
                        if node_communities
                        else "",
                        community=node_communities[0] if node_communities else None,
                        label=node_communities[0] if node_communities else None,
                        block=node_communities[0] if node_communities else None,
                    )
    
    return snapshots


def save_temporal_networkx_snapshots(
    snapshots: List,
    output_prefix: str | Path,
    format: str = "edgelist",
) -> None:
    """
    Save NetworkX temporal snapshots to files.

    Parameters
    ----------
    snapshots : List[networkx.Graph]
        List of NetworkX graph snapshots.
    output_prefix : str
        Prefix for output filenames (e.g., "graph" creates "graph_t0.txt", "graph_t1.txt", ...).
    format : str, default="edgelist"
        Output format: "edgelist", "gml", "graphml", or "gexf".
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function")
    if isinstance(output_prefix, str) is True:
        output_prefix = Path(output_prefix)

    output_prefix.mkdir(parents=True, exist_ok=True)

    for t, G in enumerate(snapshots):
        filename: Path = output_prefix / f"snapshot_t{t}.{format}"

        if format == "edgelist":
            # Save as edge list (simple format)
            nx.write_edgelist(G, filename, data=False)
        elif format == "gml":
            # Create a copy and convert tuple attributes to strings
            G_copy = _convert_tuple_attrs_to_strings(G)
            nx.write_gml(G_copy, filename)
        elif format == "graphml":
            # Create a copy and convert tuple attributes to strings
            G_copy = _convert_tuple_attrs_to_strings(G)
            nx.write_graphml(G_copy, filename)
        elif format == "gexf":
            # Create a copy and convert tuple attributes to strings
            G_copy = _convert_tuple_attrs_to_strings(G)
            nx.write_gexf(G_copy, filename)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'edgelist', 'gml', 'graphml', or 'gexf'.")

    print(
        f"Saved {len(snapshots)} snapshots with prefix '{output_prefix}' in {format} format."
    )


def _convert_tuple_attrs_to_strings(G):
    """
    Convert tuple attributes in a NetworkX graph to string representations.
    
    This is needed because GraphML, GML, and GEXF formats do not support
    tuple types as data values.
    
    Parameters
    ----------
    G : networkx.Graph
        The input graph with potential tuple attributes.
    
    Returns
    -------
    networkx.Graph
        A copy of graph with tuple attributes converted to strings.
    """
    G_copy = G.copy()
    
    # Convert node attributes
    for node, attrs in G_copy.nodes(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, tuple):
                # Convert tuple to string representation
                G_copy.nodes[node][key] = str(value)
    
    # Convert edge attributes
    for u, v, attrs in G_copy.edges(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, tuple):
                # Convert tuple to string representation
                G_copy.edges[u, v][key] = str(value)
    
    # Convert graph attributes
    for key, value in list(G_copy.graph.items()):
        if isinstance(value, tuple):
            G_copy.graph[key] = str(value)
    
    return G_copy


def load_temporal_networkx_snapshots(
    folder_path: str,
    format: str = "graphml",
    pattern: str = "*_t{index}.{format}",
) -> List:
    """
    Load all NetworkX temporal snapshots from a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing snapshot files.
        Example: "dataset/labelled-graph/my-graph/snapshots"
    format : str, default="graphml"
        File format of the snapshots: "graphml", "gml", "gexf", or "edgelist".
    pattern : str, default="*_t{index}.{format}"
        File pattern to match. Use {index} as placeholder for timestamp index.
        Default pattern matches files like "graph_t0.graphml", "graph_t1.graphml", etc.

    Returns
    -------
    snapshots : List[networkx.Graph]
        List of NetworkX graphs loaded from the folder, sorted by timestamp index.
    """
    try:
        import glob
        import os

        import networkx as nx
    except ImportError as e:
        raise ImportError(f"Required library not found: {e}")

    # Ensure folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all matching files
    pattern_path = os.path.join(folder_path, f"*.{format}")
    files = glob.glob(pattern_path)

    if not files:
        raise FileNotFoundError(f"No {format} files found in {folder_path}")

    # Extract timestamp indices and sort
    snapshot_files = []
    for filepath in files:
        filename = os.path.basename(filepath)
        # Try to extract timestamp index from filename
        # Expected format: prefix_tN.format (e.g., graph_t0.graphml)
        parts = filename.replace(f".{format}", "").split("_t")
        if len(parts) == 2 and parts[1].isdigit():
            index = int(parts[1])
            snapshot_files.append((index, filepath))

    if not snapshot_files:
        raise ValueError(
            f"No valid snapshot files found. Expected format: prefix_t0.{format}, prefix_t1.{format}, etc."
        )

    # Sort by timestamp index
    snapshot_files.sort(key=lambda x: x[0])

    # Load all snapshots
    snapshots = []
    for index, filepath in snapshot_files:
        if format == "graphml":
            G = nx.read_graphml(filepath)
        elif format == "gml":
            G = nx.read_gml(filepath)
        elif format == "gexf":
            G = nx.read_gexf(filepath)
        elif format == "edgelist":
            G = nx.read_edgelist(filepath)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'graphml', 'gml', 'gexf', or 'edgelist'.")
        snapshots.append(G)

    return snapshots


def get_communities(
    graph: Any,
    attr_name: str = "communities",
    method_name: str = "LFR",
) -> Any:
    """
    Create a cdlib NodeClustering object from a NetworkX graph with community attributes.

    Parameters
    ----------
    graph : networkx.Graph
        The NetworkX graph containing community attributes on nodes.
    attr_name : str, default="communities"
        The name of the node attribute containing community information.
        Expected format: string representation of tuple, e.g., "(1, 2)" or "(0,)".
    method_name : str, default="LFR"
        Name of the community detection method.

    Returns
    -------
    NodeClustering
        A cdlib NodeClustering object with overlap enabled.
    """

    # Get node attributes for communities
    node_communities = nx.get_node_attributes(graph, name=attr_name)

    # Build a mapping from community ID to set of nodes
    # node_communities is {node_id: "(1, 2)"}
    # We need to convert to: List[List[node_id]]
    # where communities[comm_id] = [node_id1, node_id2, ...]
    comm_to_nodes = {}
    for node_id, comm_str in node_communities.items():
        # Parse string representation of tuple back to tuple
        try:
            comm_tuple = comm_str.strip("()").split(",")
            comm_tuple = tuple(map(int, comm_tuple)) if comm_tuple[0] else ()

        except (ValueError, SyntaxError):
            comm_tuple = ()

        for comm_id in comm_tuple:
            if comm_id not in comm_to_nodes:
                comm_to_nodes[comm_id] = set()
            comm_to_nodes[comm_id].add(node_id)

    # Convert to list of lists format required by NodeClustering
    # Sort by community ID for consistency
    communities_list = []
    for comm_id in sorted(comm_to_nodes.keys()):
        communities_list.append(list(comm_to_nodes[comm_id]))

    # Create NodeClustering with overlap support
    return NodeClustering(
        communities=communities_list,
        graph=graph,
        method_name=method_name,
        overlap=True,
    )
