from typing import Dict, List, Set, Tuple

import networkx as nx
from cdlib import NodeClustering


def convert_networkx_communities_to_cdlib(
    graph: nx.Graph, communities: List[Tuple]
) -> NodeClustering:
    """
    Convert a list of communities represented as tuples of nodes into a cdlib NodeClustering object.

    Args:
        graph (nx.Graph): The NetworkX graph associated with the communities.
        communities (List[Tuple]): A list of communities, where each community is represented as a tuple of nodes.

    Returns:
        NodeClustering: A cdlib NodeClustering object containing the communities.
    """
    community_list = [list(community) for community in communities]
    return NodeClustering(communities=community_list, graph=graph)


def convert_dict_communities_to_cdlib(
    graph: nx.Graph, communities: Dict[int, Set]
) -> NodeClustering:
    """
    Convert a dictionary of communities (community_id -> set of nodes) into a cdlib NodeClustering object.

    Args:
        graph (nx.Graph): The NetworkX graph associated with the communities.
        communities (Dict[int, Set]): A dictionary mapping community IDs to sets of nodes.

    Returns:
        NodeClustering: A cdlib NodeClustering object containing the communities.
    """
    community_list = [list(community) for community in communities.values()]
    return NodeClustering(communities=community_list, graph=graph)
