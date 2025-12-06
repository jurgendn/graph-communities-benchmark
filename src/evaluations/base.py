"""Evaluation metrics for community detection."""

from cdlib import NodeClustering, evaluation
import networkx as nx


def compute_modularity(graph: nx.Graph, communities: NodeClustering) -> float:
    """Compute Newman-Girvan modularity for a community partition."""
    return communities.newman_girvan_modularity().score


def compute_nmi(communities1: NodeClustering, communities2: NodeClustering) -> float:
    """Compute Normalized Mutual Information between two community partitions."""
    return evaluation.normalized_mutual_information(communities1, communities2).score
