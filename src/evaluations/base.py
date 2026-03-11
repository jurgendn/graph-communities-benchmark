"""Evaluation metrics for community detection."""
from typing import Dict, List, Set, Union

import networkx as nx
from cdlib import NodeClustering, evaluation

DegreeInCommunityMap = Dict[Union[int, str], Dict[int, float]]
BelongingCoefficientMap = Dict[Union[int, str], Dict[int, float]]


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
        belonging_function_type (str): The function $f(\alpha_uCj, \alpha_vCj)$.
                                       Options: 'product' $(\alpha_uCj * \alpha_vCj)$ or 'average' $(0.5 * (\alpha_uCj + \alpha_vCj))$.

    Returns:
        FitnessResult: An object containing the calculated modularity Q.
    """

    G: nx.Graph = graph.to_undirected()

    m: int = G.number_of_edges()
    if m == 0:
        return 0.0

    node_degrees: Dict[Union[int, str], float] = dict(G.degree(weight="weight"))

    communities: List[List[Union[int, str]]] = clustering.communities

    nodes: List[Union[int, str]] = list(G.nodes())

    d_uCj: DegreeInCommunityMap = {}
    for u in nodes:
        d_uCj[u] = {}
        for j, Cj in enumerate(communities):
            community_nodes: Set[Union[int, str]] = set(Cj)
            degree_in_Cj: float = 0.0

            for v in community_nodes:
                if G.has_edge(u, v):
                    weight: float = G[u][v].get("weight", 1.0)
                    degree_in_Cj += weight

            d_uCj[u][j] = degree_in_Cj

    alpha_uCj: BelongingCoefficientMap = {}
    for u in nodes:
        alpha_uCj[u] = {}
        sum_d_uCk: float = sum(d_uCj[u].values())

        if sum_d_uCk == 0:
            for j in d_uCj[u]:
                alpha_uCj[u][j] = 0.0
            continue

        for j in d_uCj[u]:
            alpha_uCj[u][j] = d_uCj[u][j] / sum_d_uCk

    def f(alpha_u: float, alpha_v: float, type_str: str) -> float:
        if type_str.lower() == "product":
            return alpha_u * alpha_v
        elif type_str.lower() == "average":
            return 0.5 * (alpha_u + alpha_v)
        else:
            raise ValueError(
                f"Unknown belonging_function_type: {type_str}. Must be 'product' or 'average'."
            )

    Q_sum: float = 0.0
    two_m: float = 2.0 * m

    for j, Cj in enumerate(communities):
        community_nodes: Set[Union[int, str]] = set(Cj)

        for u in community_nodes:
            d_u: float = node_degrees.get(u, 0.0)

            alpha_u: float = alpha_uCj[u].get(j, 0.0)

            for v in community_nodes:
                alpha_v: float = alpha_uCj[v].get(j, 0.0)

                A_uv: float = G[u][v].get("weight", 1.0) if G.has_edge(u, v) else 0.0

                d_v: float = node_degrees.get(v, 0.0)
                expected_edges: float = (d_u * d_v) / two_m

                diff_term: float = A_uv - expected_edges

                f_term: float = f(alpha_u, alpha_v, belonging_function_type)

                Q_sum += diff_term * f_term

    Q: float = (1.0 / two_m) * Q_sum

    return Q

def compute_modularity(graph: nx.Graph, communities: NodeClustering) -> float:
    """Compute Newman-Girvan modularity for a community partition."""
    return communities.newman_girvan_modularity().score


def compute_nmi(communities1: NodeClustering, communities2: NodeClustering) -> float:
    """Compute Normalized Mutual Information between two community partitions."""
    return evaluation.normalized_mutual_information(communities1, communities2).score
