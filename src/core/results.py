from typing import Any, Dict, List, Set

import numpy as np
from cdlib import NodeClustering
from pydantic import BaseModel, Field, model_validator


class IntermediateResults(BaseModel):
    step: int = 0
    runtime: float = 0.0
    cdlib_modularity_overlap: float = 0.0
    customize_q0_overlap: float = 0.0
    nmi: float = 0.0
    affected_nodes: int = 0
    num_communities: int = 0

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "IntermediateResults":
        """Construct IntermediateResults dynamically from a dictionary, ignoring extra keywords."""
        filtered_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls(**filtered_data)

class MethodDynamicResults(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    runtimes: List[float] = []
    cdlib_modularity_overlap_trace: List[float] = []
    customize_q0_overlap_trace: List[float] = []
    affected_nodes: List[int] = []
    iterations_per_step: List[int] = []
    num_communities: List[int] = []
    clusterings: List[NodeClustering] = []
    nmi_trace: List[float] = []  # NMI with ground truth (if available)

    def update_intermediate_results(
        self, intermediate_results: IntermediateResults
    ):
        self.runtimes.append(intermediate_results.runtime)
        self.cdlib_modularity_overlap_trace.append(intermediate_results.cdlib_modularity_overlap)
        self.customize_q0_overlap_trace.append(
            intermediate_results.customize_q0_overlap
        )
        self.affected_nodes.append(intermediate_results.affected_nodes)
        self.num_communities.append(intermediate_results.num_communities)

    @property
    def avg_runtime(self):
        if not self.runtimes:
            return 0.0
        return np.mean(self.runtimes)
    
    @property
    def total_runtime(self) -> float:
        return sum(self.runtimes)
    
    @property
    def cdlib_modularity_overlap_stability(self) -> float:
        if len(self.cdlib_modularity_overlap_trace) < 2:
            return 0.0
        return max(self.cdlib_modularity_overlap_trace) - min(
            self.cdlib_modularity_overlap_trace
        )
    
    @property
    def customize_q0_overlap_stability(self) -> float:
        if len(self.customize_q0_overlap_trace) < 2:
            return 0.0
        return max(self.customize_q0_overlap_trace) - min(
            self.customize_q0_overlap_trace
        )

    @property
    def cdlib_modularity_overlap_range(self):
        if not self.cdlib_modularity_overlap_trace:
            return (None, None)
        return (min(self.cdlib_modularity_overlap_trace), max(self.cdlib_modularity_overlap_trace))

    @property
    def customize_q0_overlap_range(self):
        if not self.customize_q0_overlap_trace:
            return (None, None)
        return (
            min(self.customize_q0_overlap_trace),
            max(self.customize_q0_overlap_trace),
        )

    @property
    def time_steps(self) -> List[int]:
        return list(range(len(self.runtimes)))
    
    @property
    def avg_cdlib_modularity_overlap(self):
        if not self.cdlib_modularity_overlap_trace:
            return 0.0
        return np.mean(self.cdlib_modularity_overlap_trace)

    @property
    def avg_customize_q0_overlap(self):
        if not self.customize_q0_overlap_trace:
            return 0.0
        return np.mean(self.customize_q0_overlap_trace)

    @property
    def avg_nmi(self) -> float:
        """Average NMI with ground truth (if available)."""
        if not self.nmi_trace:
            return 0.0
        return np.mean(self.nmi_trace)

class NonOverlapPartitionModel(BaseModel):
    node_to_communities: Dict[Any, Set[int]] = Field(default_factory=dict)
    community_to_nodes: Dict[int, Set[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mappings(self) -> "NonOverlapPartitionModel":
        for node, communities in self.node_to_communities.items():
            for community in communities:
                if community not in self.community_to_nodes:
                    self.community_to_nodes[community] = set()
                self.community_to_nodes[community].add(node)

        for community, nodes in self.community_to_nodes.items():
            for node in nodes:
                if node not in self.node_to_communities:
                    self.node_to_communities[node] = set()
                self.node_to_communities[node].add(community)

        return self

    def change_membership(self, node: Any, new_community: int):
        old_communities = self.node_to_communities.get(node)
        if old_communities is not None:
            for old_community in old_communities:
                self.community_to_nodes[old_community].discard(node)
                if not self.community_to_nodes[old_community]:
                    del self.community_to_nodes[old_community]

        self.node_to_communities[node] = {new_community}
        if new_community not in self.community_to_nodes:
            self.community_to_nodes[new_community] = set()
        self.community_to_nodes[new_community].add(node)

    def isolate_node_to_new_community(self, node: Any):
        old_communities = self.node_to_communities.get(node)
        if old_communities is not None:
            for old_community in old_communities:
                self.community_to_nodes[old_community].discard(node)
                if not self.community_to_nodes[old_community]:
                    del self.community_to_nodes[old_community]

        new_community_id = max(self.community_to_nodes.keys(), default=-1) + 1
        self.node_to_communities[node] = {new_community_id}
        self.community_to_nodes[new_community_id] = {node}

    def get_nodes_by_community(self, community: int) -> Set[Any]:
        return self.community_to_nodes.get(community, set())

    def get_communities(self) -> Set[int]:
        return set(self.community_to_nodes.keys())


class OverlapPartitionModel(BaseModel):
    node_to_communities: Dict[Any, Set[int]] = Field(default_factory=dict)
    community_to_nodes: Dict[int, Set[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mappings(self) -> "OverlapPartitionModel":
        for node, communities in self.node_to_communities.items():
            for community in communities:
                if community not in self.community_to_nodes:
                    self.community_to_nodes[community] = set()
                self.community_to_nodes[community].add(node)

        for community, nodes in self.community_to_nodes.items():
            for node in nodes:
                if node not in self.node_to_communities:
                    self.node_to_communities[node] = set()
                self.node_to_communities[node].add(community)

        return self

    @classmethod
    def from_nx_louvain(cls, membership_dict: List[Set]) -> "OverlapPartitionModel":
        node_to_communities = {}
        community_to_nodes = {}
        for community_id, nodes in enumerate(membership_dict):
            for node in nodes:
                if node not in node_to_communities:
                    node_to_communities[node] = set()
                node_to_communities[node].add(community_id)
            if community_id not in community_to_nodes:
                community_to_nodes[community_id] = set()
            community_to_nodes[community_id].add(node)
        return cls(
            node_to_communities=node_to_communities,
            community_to_nodes=community_to_nodes,
        )

    @classmethod
    def from_community_list(
        cls, communities: List[Set[Any]]
    ) -> "OverlapPartitionModel":
        node_to_comm = {}
        comm_to_node = {}
        for cid, members in enumerate(communities):
            comm_to_node[cid] = set(members)
            for node in members:
                if node not in node_to_comm:
                    node_to_comm[node] = set()
                node_to_comm[node].add(cid)
        return cls(node_to_communities=node_to_comm, community_to_nodes=comm_to_node)

    def to_community_list(self) -> List[Set[Any]]:
        max_community_id = max(self.community_to_nodes.keys(), default=-1)
        communities = []
        for cid in range(max_community_id + 1):
            members = self.community_to_nodes.get(cid, set())
            communities.append(members)
        return communities

    def to_cdlib_node_clustering(self) -> NodeClustering:
        communities = [list(c) for c in self.to_community_list()]
        return NodeClustering(
            communities=communities,
            graph=None,
            method_name="OverlapPartitionModel",
            overlap=True,
        )

    def add_membership(self, node: Any, community: int):
        if node not in self.node_to_communities:
            self.node_to_communities[node] = set()
        self.node_to_communities[node].add(community)

        if community not in self.community_to_nodes:
            self.community_to_nodes[community] = set()
        self.community_to_nodes[community].add(node)

    def add_membership_bulk(self, node: Any, communities: Set[int]):
        if node not in self.node_to_communities:
            self.node_to_communities[node] = set()
        self.node_to_communities[node].update(communities)

        for community in communities:
            if community not in self.community_to_nodes:
                self.community_to_nodes[community] = set()
            self.community_to_nodes[community].add(node)

    def remove_membership(self, node: Any, community: int):
        if node in self.node_to_communities:
            self.node_to_communities[node].discard(community)
            if not self.node_to_communities[node]:
                del self.node_to_communities[node]

        if community in self.community_to_nodes:
            self.community_to_nodes[community].discard(node)
            if not self.community_to_nodes[community]:
                del self.community_to_nodes[community]

    def remove_membership_bulk(self, node: Any, communities: Set[int]):
        if node in self.node_to_communities:
            self.node_to_communities[node].difference_update(communities)
            if not self.node_to_communities[node]:
                del self.node_to_communities[node]

        for community in communities:
            if community in self.community_to_nodes:
                self.community_to_nodes[community].discard(node)
                if not self.community_to_nodes[community]:
                    del self.community_to_nodes[community]

    def change_membership(self, node: Any, old_community: int, new_community: int):
        self.remove_membership(node, old_community)
        self.add_membership(node, new_community)

    def change_membership_bulk(
        self, node: Any, old_communities: Set[int], new_communities: Set[int]
    ):
        self.remove_membership_bulk(node, old_communities)
        self.add_membership_bulk(node, new_communities)

    def get_nodes_by_community(self, community: int) -> Set[Any]:
        return self.community_to_nodes.get(community, set())

    def get_communities_by_node(self, node: Any) -> Set[int]:
        return self.node_to_communities.get(node, set())

    def get_communities(self) -> Set[int]:
        return set(self.community_to_nodes.keys())
