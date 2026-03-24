"""
Unified interface contract for community detection algorithms.
"""
from abc import ABC, abstractmethod
from typing import List

from cdlib import NodeClustering

from src.core.temporal_graph import TemporalGraph


class CommunityDetectionAlgorithm(ABC):
    """
    Abstract base class for all community detection algorithms.

    Every algorithm — static or dynamic — must implement ``__call__``.
    Static algorithms iterate snapshots independently; dynamic algorithms
    use the full temporal graph for incremental updates.

    Returns one ``NodeClustering`` per snapshot, in the same order as
    ``tg.iter_snapshots()``.
    """

    @abstractmethod
    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run community detection on a temporal graph.

        Args:
            tg: TemporalGraph with snapshots.

        Returns:
            List[NodeClustering], one per snapshot, in the same order
            as ``tg.iter_snapshots()``.
        """
