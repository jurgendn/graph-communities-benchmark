"""
Dynamic Algorithm Template
==========================

Copy this file to:
  src/models/dynamic/overlap/your_algorithm.py   (for overlapping)
  src/models/dynamic/crisp/your_algorithm.py     (for crisp)

Then:
  1. Rename the class (YourDynamicAlgorithm → something meaningful).
  2. Fill in __init__ parameters and the main __call__ logic.
  3. Register in config/algorithms.yaml (see bottom of this file).

A dynamic algorithm receives the full TemporalGraph (all snapshots plus
the change stream between them) and returns one NodeClustering per
snapshot. Unlike static algorithms, it can carry state across snapshots
to exploit temporal structure.
"""

from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class YourDynamicAlgorithm(CommunityDetectionAlgorithm):
    """
    One-line description of your algorithm.

    Longer description: what problem it solves, the key idea, relevant paper.
    Include how it uses temporal information (edge streams, node arrivals…).

    Args:
        param1: Description and sensible default.
        param2: Description and sensible default.
    """

    def __init__(self, param1: int = 1, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    # ------------------------------------------------------------------
    # Public interface — do NOT rename or change the signature.
    # ------------------------------------------------------------------

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run community detection over the full temporal graph.

        Unlike static algorithms, you have access to the whole change
        stream (``tg.changes``) in addition to the individual snapshots
        (``tg.iter_snapshots()``).

        Args:
            tg: TemporalGraph with snapshots and change stream.

        Returns:
            List[NodeClustering], one entry per snapshot, in the same
            order as ``tg.iter_snapshots()``.
        """
        results = []

        # ------------------------------------------------------------------
        # Example skeleton: iterate snapshots while maintaining state.
        # Replace with your incremental / streaming algorithm.
        # ------------------------------------------------------------------
        state = self._initialize_state()

        for i, snapshot in enumerate(tg.iter_snapshots()):
            # Get the edges added/removed since the previous snapshot.
            # tg.changes[i] is a TemporalChanges object with:
            #   .added_edges   — list of (u, v) tuples
            #   .removed_edges — list of (u, v) tuples
            #   .added_nodes   — list of node IDs
            #   .removed_nodes — list of node IDs
            changes = tg.changes[i] if i < len(tg.changes) else None

            # Update your internal state based on the changes.
            state = self._update_state(state, snapshot, changes)

            # Detect communities from the current state.
            communities = self._detect_communities(state, snapshot)

            nc = NodeClustering(
                communities=communities,
                graph=snapshot,
                method_name="YourDynamicAlgorithm",
            )
            results.append(nc)

        return results

    # ------------------------------------------------------------------
    # Private helpers — rename and add as needed.
    # ------------------------------------------------------------------

    def _initialize_state(self) -> dict:
        """
        Create the initial algorithm state (before the first snapshot).

        Returns whatever structure your algorithm uses to track community
        memberships across time steps (e.g. a dict mapping node → community).
        """
        return {}

    def _update_state(self, state: dict, snapshot: nx.Graph, changes) -> dict:
        """
        Incrementally update internal state based on new graph changes.

        Args:
            state:    Current algorithm state.
            snapshot: The full graph at this time step.
            changes:  TemporalChanges for this step (may be None for step 0).

        Returns:
            Updated state.
        """
        # TODO: apply added/removed edges and nodes to your state.
        return state

    def _detect_communities(self, state: dict, graph: nx.Graph) -> List[List]:
        """
        Extract community assignments from the current state.

        Args:
            state:  Current algorithm state.
            graph:  The full graph at this time step.

        Returns:
            List of communities, where each community is a list of node IDs.
            Nodes may appear in multiple communities (overlapping) or exactly
            one (crisp).
        """
        raise NotImplementedError("Replace this with your community extraction logic.")


# ----------------------------------------------------------------------
# config/algorithms.yaml registration snippet
# ----------------------------------------------------------------------
#
# Add the following block under `temporal_algorithms:` in config/algorithms.yaml,
# then add your algorithm name to `target_temporal_algorithms:` to enable it.
#
# your_dynamic_algorithm:
#   module: "src.models.dynamic.overlap.your_algorithm"  # adjust path
#   function: "YourDynamicAlgorithm"
#   params:
#     param1: 1
#     param2: 0.5
#   clustering_type: "overlapping"   # or "crisp"
#   metadata: {}
#   description: "Short description of your dynamic algorithm"
