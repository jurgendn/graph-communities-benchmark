"""
Visualization utilities for overlapping community detection.

This module provides a Visualizer class to visualize overlapping communities in graphs
using a custom circular layout where overlapping nodes are positioned between
their respective communities.
"""

from typing import Dict, List, Optional, Tuple

# Import community detection algorithms
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cdlib import NodeClustering, algorithms
from matplotlib.lines import Line2D


def generate_sample_graph(n: int = 40, s: int = 15, v: int = 2,
                          p_in: float = 0.7, p_out: float = 0.1,
                          directed: bool = False) -> nx.Graph:
    """
    Generate a sample graph with Gaussian random partition structure.

    Parameters
    ----------
    n : int, default=40
        Number of nodes in the graph.
    s : int, default=15
        Size of each cluster.
    v : int, default=2
        Number of clusters.
    p_in : float, default=0.7
        Probability of edges within clusters.
    p_out : float, default=0.1
        Probability of edges between clusters.
    directed : bool, default=False
        Whether the graph is directed.

    Returns
    -------
    networkx.Graph
        The generated graph.
    """
    return nx.gaussian_random_partition_graph(
        n=n, s=s, v=v, p_in=p_in, p_out=p_out, directed=directed
    )


def detect_overlapping_communities(graph: nx.Graph, algorithm: str = "angel",
                                    **kwargs) -> NodeClustering:
    """
    Detect overlapping communities in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.
    algorithm : str, default="angel"
        Name of the overlapping community detection algorithm.
        Currently supports "angel".
    **kwargs
        Additional arguments to pass to the algorithm.

    Returns
    -------
    NodeClustering
        NodeClustering object containing the detected communities.
    """
    if algorithm == "angel":
        threshold = kwargs.get("threshold", 0.25)
        return algorithms.angel(graph, threshold=threshold)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class Visualizer:
    """
    A class for visualizing overlapping communities in graphs.

    This visualizer uses a custom circular layout where overlapping nodes
    are positioned between their respective communities.
    """

    def __init__(self, radius_outer: float = 3.0, radius_inner: float = 0.6,
                 cmap_name: str = "tab20", seed: Optional[int] = None):
        """
        Initialize the Visualizer.

        Parameters
        ----------
        radius_outer : float, default=3.0
            Radius for cluster centers in the circular layout.
        radius_inner : float, default=0.6
            Radius for nodes within each cluster.
        cmap_name : str, default="tab20"
            Name of the colormap to use for communities.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.radius_outer = radius_outer
        self.radius_inner = radius_inner
        self.cmap_name = cmap_name
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def calculate_membership_counts(graph: nx.Graph,
                                     communities: List[List[int]]) -> Dict[int, int]:
        """
        Calculate the number of communities each node belongs to.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph.
        communities : list of list
            List of communities.

        Returns
        -------
        dict
            Dictionary mapping nodes to their membership counts.
        """
        return {n: sum(n in c for c in communities) for n in graph.nodes}

    @staticmethod
    def get_overlap_nodes(membership_counts: Dict[int, int]) -> List[int]:
        """
        Get nodes that belong to more than one community.

        Parameters
        ----------
        membership_counts : dict
            Dictionary mapping nodes to their membership counts.

        Returns
        -------
        list
            List of overlapping nodes.
        """
        return [n for n, cnt in membership_counts.items() if cnt > 1]

    def compute_circular_layout(self, graph: nx.Graph, communities: List[List[int]],
                                overlap_nodes: Optional[List[int]] = None,
                                radius_outer: Optional[float] = None,
                                radius_inner: Optional[float] = None,
                                seed: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
        """
        Compute a circular layout for communities with overlapping nodes between clusters.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph.
        communities : list of list
            List of communities.
        overlap_nodes : list, optional
            List of overlapping nodes. If None, will be computed.
        radius_outer : float, optional
            Radius for cluster centers. Uses instance default if None.
        radius_inner : float, optional
            Radius for nodes within each cluster. Uses instance default if None.
        seed : int, optional
            Random seed for reproducibility. Uses instance default if None.

        Returns
        -------
        dict
            Dictionary mapping nodes to their (x, y) positions.
        """
        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)

        if radius_outer is None:
            radius_outer = self.radius_outer
        if radius_inner is None:
            radius_inner = self.radius_inner

        if overlap_nodes is None:
            membership_counts = self.calculate_membership_counts(graph, communities)
            overlap_nodes = self.get_overlap_nodes(membership_counts)

        n_communities = len(communities)
        pos = {}

        # Place cluster centers in a circle
        cluster_centers = []
        for i in range(n_communities):
            angle = 2 * np.pi * i / n_communities
            cx = radius_outer * np.cos(angle)
            cy = radius_outer * np.sin(angle)
            cluster_centers.append((cx, cy))

        # Place nodes around their cluster centers
        for node in graph.nodes:
            if node in overlap_nodes:
                # Position overlapping nodes between their clusters
                member_idxs = [i for i, c in enumerate(communities) if node in c]
                if len(member_idxs) >= 2:
                    # Place between first two communities
                    cx1, cy1 = cluster_centers[member_idxs[0]]
                    cx2, cy2 = cluster_centers[member_idxs[1]]
                    # Midpoint with slight offset
                    blend = 0.5 + 0.1 * (np.random.random() - 0.5)
                    px = blend * cx1 + (1 - blend) * cx2
                    py = blend * cy1 + (1 - blend) * cy2
                    pos[node] = (px, py)
                else:
                    # Fallback to first cluster
                    cx, cy = cluster_centers[member_idxs[0]]
                    pos[node] = (cx, cy)
            else:
                # Place non-overlapping nodes around their cluster center
                comm_idx = next((i for i, c in enumerate(communities) if node in c), None)
                if comm_idx is not None:
                    cx, cy = cluster_centers[comm_idx]
                    angle = 2 * np.pi * np.random.random()
                    r = radius_inner * np.sqrt(np.random.random())
                    pos[node] = (cx + r * np.cos(angle), cy + r * np.sin(angle))
                else:
                    # Node not in any community - place at origin
                    pos[node] = (0, 0)

        return pos

    def visualize(self, graph: nx.Graph, communities: NodeClustering,
                  pos: Optional[Dict[int, Tuple[float, float]]] = None,
                  title: str = "Overlapping communities - Circular layout with overlap nodes between clusters",
                  figsize: Tuple[int, int] = (10, 10),
                  cmap_name: Optional[str] = None,
                  show_legend: bool = True,
                  seed: Optional[int] = None) -> plt.Figure:
        """
        Visualize overlapping communities with a circular layout.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph.
        communities : NodeClustering
            NodeClustering object containing the detected communities.
        pos : dict, optional
            Pre-computed node positions. If None, will be computed.
        title : str, default="Overlapping communities..."
            Title for the plot.
        figsize : tuple, default=(10, 10)
            Figure size (width, height).
        cmap_name : str, optional
            Name of the colormap to use for communities. Uses instance default if None.
        show_legend : bool, default=True
            Whether to show the legend.
        seed : int, optional
            Random seed for layout computation. Uses instance default if None.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)

        if cmap_name is None:
            cmap_name = self.cmap_name

        # Extract communities list from NodeClustering object
        communities_list = communities.communities

        # Calculate membership counts and overlap nodes
        membership_counts = self.calculate_membership_counts(graph, communities_list)
        overlap_nodes = self.get_overlap_nodes(membership_counts)

        # Compute layout if not provided
        if pos is None:
            pos = self.compute_circular_layout(graph, communities_list, overlap_nodes, seed=seed)

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.2, width=0.5)

        cmap = plt.cm.get_cmap(cmap_name)
        legend_handles = []

        # Draw community nodes
        for idx, comm in enumerate(communities_list):
            color = cmap(idx % cmap.N)
            non_overlap_comm = [n for n in comm if n not in overlap_nodes]
            if non_overlap_comm:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=non_overlap_comm,
                    node_color=[color],
                    node_size=150,
                    alpha=0.7,
                )
            if idx < 8:
                legend_handles.append(
                    Line2D(
                        [0], [0], marker="o", color="w", label=f"Comm {idx + 1}",
                        markerfacecolor=color, markersize=10, markeredgecolor="none"
                    )
                )

        # Draw overlapping nodes with distinct styling
        if overlap_nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=overlap_nodes,
                node_color="white", edgecolors="red", linewidths=3,
                node_size=300, alpha=1.0, zorder=10
            )
            nx.draw_networkx_labels(
                graph, pos,
                labels={n: str(membership_counts[n]) for n in overlap_nodes},
                font_size=10, font_color="red", font_weight="bold"
            )
            legend_handles.append(
                Line2D(
                    [0], [0], marker="o", color="w", label="Overlap node (#comms)",
                    markerfacecolor="none", markeredgecolor="red", markersize=12, markeredgewidth=3
                )
            )

        plt.axis("off")
        plt.title(title, fontsize=14)
        if show_legend:
            plt.legend(handles=legend_handles, loc="upper left", frameon=False)
        plt.tight_layout()

        return plt.gcf()

    def demo(self, n: int = 40, s: int = 15, v: int = 2,
             p_in: float = 0.7, p_out: float = 0.1,
             threshold: float = 0.25, seed: Optional[int] = None):
        """
        Demonstrate the visualization with a sample graph.

        Parameters
        ----------
        n : int, default=40
            Number of nodes.
        s : int, default=15
            Size of each cluster.
        v : int, default=2
            Number of clusters.
        p_in : float, default=0.7
            Probability of edges within clusters.
        p_out : float, default=0.1
            Probability of edges between clusters.
        threshold : float, default=0.25
            Threshold for ANGEL algorithm.
        seed : int, optional
            Random seed for reproducibility. Uses instance default if None.

        Returns
        -------
        tuple
            (graph, communities) where graph is the generated networkx.Graph
            and communities is the NodeClustering object.
        """
        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)

        # Generate sample graph
        graph = generate_sample_graph(n=n, s=s, v=v, p_in=p_in, p_out=p_out)

        # Detect overlapping communities
        communities = detect_overlapping_communities(graph, algorithm="angel", threshold=threshold)

        # Visualize
        fig = self.visualize(graph, communities, seed=seed)
        plt.show()

        return graph, communities


# Main execution block for backward compatibility
if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.demo()
