"""
Static Algorithm Template
=========================

Copy this file to:
  src/models/static_methods/overlap/your_algorithm.py   (for overlapping)
  src/models/static_methods/crisp/your_algorithm.py     (for crisp)

Then:
  1. Rename the class (YourStaticAlgorithm → something meaningful).
  2. Fill in __init__ parameters and _detect_communities logic.
  3. Register in config/algorithms.yaml (see bottom of this file).

A static algorithm is applied independently to each snapshot of the
temporal graph. It receives one networkx Graph and returns one
NodeClustering.
"""

from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class YourStaticAlgorithm(CommunityDetectionAlgorithm):
    """
    One-line description of your algorithm.

    Longer description: what problem it solves, the key idea, relevant paper.

    Args:
        param1: Description and sensible default.
        param2: Description and sensible default.
    """

    def __init__(self, param1: float = 0.5, param2: int = 10):
        self.param1 = param1
        self.param2 = param2

    # ------------------------------------------------------------------
    # Public interface — do NOT rename or change the signature.
    # ------------------------------------------------------------------

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run community detection on every snapshot of a temporal graph.

        Args:
            tg: TemporalGraph containing graph snapshots.

        Returns:
            List[NodeClustering], one entry per snapshot, in the same
            order as ``tg.iter_snapshots()``.
        """
        results = []
        for snapshot in tg.iter_snapshots():
            communities = self._detect_communities(snapshot)
            nc = NodeClustering(
                communities=communities,
                graph=snapshot,
                method_name="YourStaticAlgorithm",
            )
            results.append(nc)
        return results

    # ------------------------------------------------------------------
    # Private helpers — rename and add as needed.
    # ------------------------------------------------------------------

    def _detect_communities(self, graph: nx.Graph) -> List[List]:
        """
        Detect communities in a single graph snapshot.

        Args:
            graph: A networkx Graph for one time step.

        Returns:
            List of communities, where each community is a list of node IDs.
            Nodes may appear in multiple communities (overlapping) or exactly
            one (crisp).

        Example (random partition — replace with real logic):
            nodes = list(graph.nodes())
            mid = len(nodes) // 2
            return [nodes[:mid], nodes[mid:]]
        """
        raise NotImplementedError("Replace this with your community detection logic.")


# ----------------------------------------------------------------------
# config/algorithms.yaml registration snippet
# ----------------------------------------------------------------------
#
# Add the following block under `algorithms:` in config/algorithms.yaml,
# then add your algorithm name to `target_algorithms:` to enable it.
#
# your_algorithm:
#   module: "src.models.static_methods.overlap.your_algorithm"  # adjust path
#   function: "YourStaticAlgorithm"
#   params:
#     param1: 0.5
#     param2: 10
#   type: "static"
#   clustering_type: "overlapping"   # or "crisp"
#   metadata: {}
#   description: "Short description of your algorithm"
