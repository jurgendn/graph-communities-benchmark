import time

import networkx as nx
from cdlib import NamedClustering, NodeClustering, TemporalClustering, algorithms, evaluation
from cdlib.algorithms.internal_dcd.eTILES import eTILES
from dynetx import DynGraph

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.evaluation.target_modularity import overlapping_modularity_q0
from src.core.results import IntermediateResults, MethodDynamicResults
from src.core.temporal_graph import TemporalGraph


def _patched_tiles(dg: object, obs: int = 1) -> TemporalClustering:
    """
    Patched version of cdlib.tiles that handles empty lifecycle lists.
    """
    alg = eTILES(dg=dg, obs=obs)
    tc = TemporalClustering()
    t = obs
    for c in alg.execute():
        communities = {}
        for k, v in c.items():
            communities[f"{t}_{k}"] = v
        sg = dg.time_slice(t - obs, t)

        nc = NamedClustering(communities, sg, "TILES", {"obs": obs}, overlap=True)

        if t <= max(dg.temporal_snapshots_ids()):
            tc.add_clustering(nc, time=t)
            t += obs
        else:
            break

    # add community matches (can contain communities not present in the observations)
    mtc = alg.get_matches()
    tc.add_matching(mtc)

    ### polytree

    # cleaning & updating community matching
    # PATCHED: Check if lifecycle is not empty before accessing lifecycle[0]
    dg = _patched_lifecycle_polytree(tc)
    community_ids = list(dg.nodes())

    tids = tc.get_observation_ids()
    ndss = []
    for tid in tids:
        c = tc.get_clustering_at(tid)
        comunity_ids_m = c.named_communities.keys()
        for ci in comunity_ids_m:
            ndss.append(ci)

    for tid in community_ids:
        if tid not in ndss:
            dg.remove_node(tid)

    mtc = list(dg.edges())
    tc.add_matching(mtc)

    return tc


def _patched_lifecycle_polytree(tc) -> nx.DiGraph:
    """
    Reconstruct the poly-tree representing communities lifecycles using a provided similarity function.
    Patched version to handle empty lifecycle lists.
    """
    lifecycle = tc.matching

    pt = nx.DiGraph()
    # Check if lifecycle is not empty before accessing lifecycle[0]
    if lifecycle and len(lifecycle[0]) == 3:
        for u, v, w in lifecycle:
            pt.add_edge(u, v, weight=w)
    elif lifecycle:
        # implicit matching
        for u, v in lifecycle:
            pt.add_edge(u, v)

    return pt


# Apply the monkey patch to algorithms.tiles
algorithms.tiles = _patched_tiles


def _named_to_nodeclustering(named: NamedClustering, graph: nx.Graph) -> NodeClustering:
    """
    Convert a cdlib.NamedClustering instance to a NodeClustering on the given graph.
    """
    communities_list = [list(comm) for comm in named.communities if comm]
    return NodeClustering(
        communities=communities_list,
        graph=graph,
        method_name=getattr(named, "method_name", "TILES"),
        method_parameters=getattr(named, "method_parameters", {}),
        overlap=True,
    )


@register(
    name="tiles",
    algo_type="dynamic",
    clustering_type="overlapping",
    default_params={"obs": 1},
    description="TILES: Temporal Incremental Link-based Evolutionary System",
)
class Tiles(CommunityDetectionAlgorithm):
    def __init__(self, obs: int = 1):
        self.obs = obs

    def run(self, tg: TemporalGraph) -> MethodDynamicResults:
        """
        Apply the TILES algorithm to a temporal graph.

        Args:
            tg: A dynamic graph (DynGraph) representing the temporal network.

        Returns:
            A list of TemporalClustering objects, one for each snapshot.
        """
        dgraph = DynGraph()
        for t, snapshot in enumerate(tg.iter_snapshots()):
            dgraph.add_interactions_from(list(snapshot.edges()), t=t)
        start_time = time.perf_counter()
        res = algorithms.tiles(dg=dgraph, obs=self.obs)
        elapsed = time.perf_counter() - start_time
        results = MethodDynamicResults()
        for snapshot, community in zip(tg.iter_snapshots(), res.clusterings.values()):
            # Convert NamedClustering to NodeClustering for a uniform interface
            node_clustering = _named_to_nodeclustering(community, snapshot)
            # Guard against empty communities to avoid ZeroDivisionError in cdlib
            if node_clustering.communities:
                q = evaluation.modularity_overlap(snapshot, node_clustering).score
            else:
                q = 0.0
            q0 = overlapping_modularity_q0(snapshot, node_clustering)
            current_res = IntermediateResults(
                runtime=elapsed,
                cdlib_modularity_overlap=q,
                customize_q0_overlap=q0,
                affected_nodes=snapshot.number_of_nodes(),
                num_communities=len(node_clustering.communities),
            )
            results.update_intermediate_results(current_res)
            results.clusterings.append(node_clustering)
        return results

    def __call__(self, tg: TemporalGraph):
        """Run TILES on the temporal graph.

        Returns:
            MethodDynamicResults for the unified benchmark pipeline.
        """
        return self.run(tg)
