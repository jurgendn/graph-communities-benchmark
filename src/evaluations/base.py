"""Evaluation metrics for community detection."""
from typing import Callable, Dict, List, Set, Union

import networkx as nx
from cdlib import NodeClustering, evaluation
from cdlib.classes import NodeClustering


# Define type aliases for complex dictionaries used
# d_uCj: Dict[node_id, Dict[community_index, degree_in_community]]
DegreeInCommunityMap = Dict[Union[int, str], Dict[int, float]]
# alpha_uCj: Dict[node_id, Dict[community_index, belonging_coefficient]]
BelongingCoefficientMap = Dict[Union[int, str], Dict[int, float]]
# Adjacency Term: Union[int, float]


def generalized_overlapping_modularity(
    graph: nx.Graph,
    clustering: NodeClustering,
    belonging_function_type: str = "average",
):
    """
    Calculates the generalized modularity Q for overlapping clustering
    as defined by Chen et al. [6] (Eq. 5.1 and Eq. 1.2).

    Args:
        graph (nx.Graph): The networkx graph object.
        clustering (NodeClustering): The community structure found by an algorithm.
        belonging_function_type (str): The function f(α_uCj, α_vCj).
                                       Options: 'Average' (α_uCj * α_vCj) or 'average' (0.5 * (α_uCj + α_vCj)).

    Returns:
        FitnessResult: An object containing the calculated modularity Q.
    """

    # 1. Preliminaries
    G: nx.Graph = graph.to_undirected()

    # Total number of edges * 2
    m: int = G.number_of_edges()
    if m == 0:
        return 0.0

    # Dictionary to store node degrees: d_u
    node_degrees: Dict[Union[int, str], float] = dict(G.degree(weight="weight"))

    # Get the list of communities C_j from the NodeClustering object
    communities: List[List[Union[int, str]]] = clustering.communities

    # The set of all nodes V
    nodes: List[Union[int, str]] = list(G.nodes())

    # 2. Calculate the intermediate term d_uCj for all nodes u and communities Cj
    # d_uCj = Sum_{v in Cj} A_uv (Eq. 1.2 definition)
    d_uCj: DegreeInCommunityMap = {}
    for u in nodes:
        d_uCj[u] = {}
        for j, Cj in enumerate(communities):
            community_nodes: Set[Union[int, str]] = set(Cj)
            degree_in_Cj: float = 0.0

            # Sum A_uv for all v in Cj
            for v in community_nodes:
                if G.has_edge(u, v):
                    # Use 'weight' attribute if available, otherwise assume 1.0 (unweighted)
                    weight: float = G[u][v].get("weight", 1.0)
                    degree_in_Cj += weight

            d_uCj[u][j] = degree_in_Cj

    # 3. Calculate the Belonging Coefficient α_uCj (Eq. 1.2)
    # α_uCj = d_uCj / Sum_{Ck in C} d_uCk
    alpha_uCj: BelongingCoefficientMap = {}
    for u in nodes:
        alpha_uCj[u] = {}
        # Denominator: Sum_{Ck in C} d_uCk over all communities Ck
        sum_d_uCk: float = sum(d_uCj[u].values())

        # Handle the case where a node has no connection to any detected community
        if sum_d_uCk == 0:
            for j in d_uCj[u]:
                alpha_uCj[u][j] = 0.0
            continue

        for j in d_uCj[u]:
            alpha_uCj[u][j] = d_uCj[u][j] / sum_d_uCk

    def f(alpha_u: float, alpha_v: float, type_str: str) -> float:
        """Helper function to calculate f(α_uCj, α_vCj)."""
        if type_str.lower() == "product":
            # f = α_uCj * α_vCj
            return alpha_u * alpha_v
        elif type_str.lower() == "average":
            # f = 0.5 * (α_uCj + α_vCj)
            return 0.5 * (alpha_u + alpha_v)
        else:
            raise ValueError(
                f"Unknown belonging_function_type: {type_str}. Must be 'product' or 'average'."
            )

    # 5. Calculate the Generalized Modularity Q (Eq. 5.1)
    # Q = (1 / 2m) * Sum_{Cj in C} Sum_{u,v in Cj} (A_uv - (d_u * d_v) / 2m) * f(α_uCj, α_vCj)

    Q_sum: float = 0.0
    two_m: float = 2.0 * m

    # Outer sum: Sum_{Cj in C}
    for j, Cj in enumerate(communities):
        community_nodes: Set[Union[int, str]] = set(Cj)

        # Inner sum: Sum_{u,v in Cj} over all ordered pairs (u, v) in the community
        for u in community_nodes:
            d_u: float = node_degrees.get(u, 0.0)

            # Retrieve α_uCj for the current community j
            alpha_u: float = alpha_uCj[u].get(j, 0.0)

            for v in community_nodes:
                # Retrieve α_vCj for the current community j
                alpha_v: float = alpha_uCj[v].get(j, 0.0)

                # A_uv: Adjacency term (edge weight or 1 if unweighted)
                A_uv: float = G[u][v].get("weight", 1.0) if G.has_edge(u, v) else 0.0

                # Degree term: (d_u * d_v) / 2m
                d_v: float = node_degrees.get(v, 0.0)
                expected_edges: float = (d_u * d_v) / two_m

                # Difference term
                diff_term: float = A_uv - expected_edges

                # Belonging function term: f(α_uCj, α_vCj)
                f_term: float = f(alpha_u, alpha_v, belonging_function_type)

                # Full term for this (u, v, Cj)
                Q_sum += diff_term * f_term

    # Final Modularity Q
    Q: float = (1.0 / two_m) * Q_sum

    # Return as a cdlib.evaluation.FitnessResult object
    return Q

def compute_modularity(graph: nx.Graph, communities: NodeClustering) -> float:
    """Compute Newman-Girvan modularity for a community partition."""
    return communities.newman_girvan_modularity().score


def compute_nmi(communities1: NodeClustering, communities2: NodeClustering) -> float:
    """Compute Normalized Mutual Information between two community partitions."""
    return evaluation.normalized_mutual_information(communities1, communities2).score
