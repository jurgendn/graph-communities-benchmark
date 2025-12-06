from typing import Any, Dict, List, Set

import numpy as np
from pydantic import BaseModel, Field, model_validator


class IntermediateResults(BaseModel):
    runtime: float = 0.0
    modularity: float = 0.0
    affected_nodes: int = 0
    num_communities: int = 0

class MethodDynamicResults(BaseModel):
    runtimes: List[float] = []
    modularities: List[float] = []
    affected_nodes: List[int] = []
    iterations_per_step: List[int] = []
    num_communities: List[int] = []

    def update_intermediate_results(
        self, intermediate_results: IntermediateResults
    ):
        self.runtimes.append(intermediate_results.runtime)
        self.modularities.append(intermediate_results.modularity)
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
    def modularity_stability(self) -> float:
        if len(self.modularities) < 2:
            return 0.0
        return max(self.modularities) - min(self.modularities)
    
    @property
    def modularity_range(self):
        if not self.modularities:
            return (None, None)
        return (min(self.modularities), max(self.modularities))

    @property
    def time_steps(self) -> List[int]:
        return list(range(len(self.runtimes)))
    
    @property
    def avg_modularities(self):
        if not self.modularities:
            return 0.0
        return np.mean(self.modularities)
