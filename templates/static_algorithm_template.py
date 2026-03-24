"""
Static Algorithm Template
=========================

Copy this file to:
  src/models/static/overlap/your_algorithm.py   (for overlapping)
  src/models/static/crisp/your_algorithm.py     (for crisp)

Then:
  1. Rename the class.
  2. Fill in ``_detect_communities`` with your logic.
  3. Update the ``@register(...)`` decorator with your algorithm's name and metadata.
  4. Add the algorithm name to ``target_snapshot_algorithms`` in config/algorithms.yaml.
  5. Add the module path to ``_REGISTRATION_MODULES`` in src/algorithms/factory.py.

A static algorithm is applied independently to each snapshot of the
temporal graph.  It receives one ``networkx.Graph`` per snapshot and must
produce one ``cdlib.NodeClustering`` per snapshot.

Integration checklist (3 touches):
  - This file  (implement + @register)
  - config/algorithms.yaml  (add name to target list, optionally override params)
  - src/algorithms/factory.py  (add module to _REGISTRATION_MODULES)
"""

from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.core.temporal_graph import TemporalGraph


@register(
    name="your_static_algorithm",          # unique name, must match YAML reference
    algo_type="static",
    clustering_type="overlapping",         # or "crisp"
    default_params={
        "param1": 0.5,
        "param2": 10,
    },
    description="Short description of your algorithm",
)
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
    # Public interface -- do NOT rename or change the signature.
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
    # Private helpers -- rename and add as needed.
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

        Example (random partition -- replace with real logic):
            nodes = list(graph.nodes())
            mid = len(nodes) // 2
            return [nodes[:mid], nodes[mid:]]
        """
        raise NotImplementedError("Replace this with your community detection logic.")


# ----------------------------------------------------------------------
# config/algorithms.yaml -- add the algorithm name to the target list
# and optionally override default params:
#
# target_snapshot_algorithms:
#   - your_static_algorithm
#
# algorithm_params:               # optional
#   your_static_algorithm:
#     param1: 0.7
#     param2: 20
# ----------------------------------------------------------------------
#
# src/algorithms/factory.py -- add the module path to _REGISTRATION_MODULES:
#
# _REGISTRATION_MODULES = [
#     ...
#     "src.models.static.overlap.your_algorithm",
# ]
