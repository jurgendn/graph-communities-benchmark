from typing import List

import numpy as np
from pydantic import BaseModel


class IntermediateResults(BaseModel):
    runtime: float = 0.0
    cdlib_modularity_overlap: float = 0.0
    customize_q0_overlap: float = 0.0
    affected_nodes: int = 0
    num_communities: int = 0

class MethodDynamicResults(BaseModel):
    runtimes: List[float] = []
    cdlib_modularity_overlap_trace: List[float] = []
    customize_q0_overlap_trace: List[float] = []
    affected_nodes: List[int] = []
    iterations_per_step: List[int] = []
    num_communities: List[int] = []

    def update_intermediate_results(
        self, intermediate_results: IntermediateResults
    ):
        self.runtimes.append(intermediate_results.runtime)
        self.cdlib_modularity_overlap_trace.append(intermediate_results.cdlib_modularity_overlap)
        self.customize_q0_overlap_trace.append(intermediate_results.customize_q0_overlap)
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