"""
Wrappers that adapt existing algorithm functions/classes to the unified
CommunityDetectionAlgorithm interface.
"""
from typing import Callable, List

from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.communities import MethodDynamicResults
from src.factory.factory import TemporalGraph


class StaticMethodWrapper(CommunityDetectionAlgorithm):
    """
    Wraps any per-snapshot function (e.g. cdlib functions or callable
    class instances) so it conforms to the unified interface.

    The wrapped function must accept a networkx Graph as its first
    positional argument and return a NodeClustering.

    Args:
        func: Callable that takes (graph, **params) → NodeClustering.
        params: Keyword arguments forwarded to ``func`` on every call.
    """

    def __init__(self, func: Callable, params: dict):
        self.func = func
        self.params = params

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        results = []
        for snapshot in tg.iter_snapshots():
            communities: NodeClustering = self.func(snapshot, **self.params)
            results.append(communities)
        return results


class DynamicMethodWrapper(CommunityDetectionAlgorithm):
    """
    Wraps dynamic algorithms that consume a full TemporalGraph rather
    than individual snapshots.

    Dynamic methods (e.g. ``run_df_louvain``) may return either:
    - MethodDynamicResults (with clusterings and metrics)
    - List[NodeClustering] (new behavior after pipeline refactoring)
    
    This wrapper normalizes output to always return List[NodeClustering].
    If MethodDynamicResults is returned, runtime info is preserved in the
    wrapper for the pipeline to use.

    Args:
        func: Callable that takes (tg, **params) → MethodDynamicResults or List[NodeClustering].
        params: Keyword arguments forwarded to ``func``.
    """

    def __init__(self, func: Callable, params: dict):
        self.func = func
        self.params = params
        self._runtime = None  # Preserved runtime for dynamic methods

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        result = self.func(tg, **self.params)
        
        # Normalize to List[NodeClustering]
        if isinstance(result, MethodDynamicResults):
            # Preserve runtime for the pipeline
            self._runtime = result.runtimes
            return result.clusterings
        
        # Already a list
        return result
