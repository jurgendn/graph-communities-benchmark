from typing import Dict, List, Set, Tuple

import networkx as nx
from cdlib import NodeClustering


def convert_networkx_communities_to_cdlib(
    graph: nx.Graph, communities: List[Tuple]
) -> NodeClustering:
    community_list = [list(community) for community in communities]
    return NodeClustering(communities=community_list, graph=graph)


def convert_dict_communities_to_cdlib(
    graph: nx.Graph, communities: Dict[int, Set]
) -> NodeClustering:
    community_list = [list(community) for community in communities.values()]
    return NodeClustering(communities=community_list, graph=graph)
