"""
Dynamic Algorithm Template
==========================

Copy this file to:
  src/models/dynamic/overlap/your_algorithm.py   (for overlapping)
  src/models/dynamic/crisp/your_algorithm.py     (for crisp)

Then:
  1. Rename the class.
  2. Fill in the ``__call__`` logic — use snapshots and ``tg.steps``.
  3. Update the ``@register(...)`` decorator with your algorithm's name and metadata.
  4. Add the algorithm name to ``target_temporal_algorithms`` in config/algorithms.yaml.
  5. Add the module path to ``_REGISTRATION_MODULES`` in src/algorithms/factory.py.

A dynamic algorithm receives the full ``TemporalGraph`` (all snapshots
plus the change stream between them) and returns one ``NodeClustering``
per snapshot.  Unlike static algorithms, it can carry state across
snapshots to exploit temporal structure.

Temporal data model:
  - ``tg.base_graph``        — the graph at t=0
  - ``tg.steps``             — list of TemporalChanges objects
  - ``tg.steps[i]``          — changes applied to get snapshot i+1
  - ``tg.steps[i].insertions`` — list of (u, v) edge tuples added
  - ``tg.steps[i].deletions``  — list of (u, v) edge tuples removed
  - ``tg.iter_snapshots()``  — yields snapshot graphs in order (t=0 … t=T)
  - ``tg[i]``                — snapshot graph at time i
  - ``len(tg)``              — total number of snapshots (len(steps) + 1)

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
    name="your_dynamic_algorithm",         # unique name, must match YAML reference
    algo_type="dynamic",
    clustering_type="overlapping",         # or "crisp"
    default_params={
        "param1": 1,
        "param2": 0.5,
    },
    description="Short description of your dynamic algorithm",
)
class YourDynamicAlgorithm(CommunityDetectionAlgorithm):
    """
    One-line description of your algorithm.

    Longer description: what problem it solves, the key idea, relevant paper.
    Include how it uses temporal information (edge streams, node arrivals...).

    Args:
        param1: Description and sensible default.
        param2: Description and sensible default.
    """

    def __init__(self, param1: int = 1, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    # ------------------------------------------------------------------
    # Public interface -- do NOT rename or change the signature.
    # ------------------------------------------------------------------

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run community detection over the full temporal graph.

        Unlike static algorithms, you have access to the change stream
        (``tg.steps``) in addition to the individual snapshots
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
            # Get the edges inserted/deleted since the previous snapshot.
            #
            # tg.steps[i-1] is a TemporalChanges object with:
            #   .insertions — list of (u, v) edge tuples
            #   .deletions  — list of (u, v) edge tuples
            #
            # At i == 0, there are no prior changes (this is the base graph).
            if i > 0:
                changes = tg.steps[i - 1]
                # changes.insertions — edges added since previous snapshot
                # changes.deletions  — edges removed since previous snapshot
            else:
                changes = None

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
    # Private helpers -- rename and add as needed.
    # ------------------------------------------------------------------

    def _initialize_state(self) -> dict:
        """
        Create the initial algorithm state (before the first snapshot).

        Returns whatever structure your algorithm uses to track community
        memberships across time steps (e.g. a dict mapping node -> community).
        """
        return {}

    def _update_state(self, state: dict, snapshot: nx.Graph, changes) -> dict:
        """
        Incrementally update internal state based on new graph changes.

        Args:
            state:    Current algorithm state.
            snapshot: The full graph at this time step.
            changes:  TemporalChanges for this step (None for step 0).
                      Access changes.insertions and changes.deletions.

        Returns:
            Updated state.
        """
        # TODO: apply inserted/deleted edges to your state.
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
# config/algorithms.yaml -- add the algorithm name to the target list
# and optionally override default params:
#
# target_temporal_algorithms:
#   - your_dynamic_algorithm
#
# algorithm_params:               # optional
#   your_dynamic_algorithm:
#     param1: 2
#     param2: 0.8
# ----------------------------------------------------------------------
#
# src/algorithms/factory.py -- add the module path to _REGISTRATION_MODULES:
#
# _REGISTRATION_MODULES = [
#     ...
#     "src.models.dynamic.overlap.your_algorithm",
# ]
