from typing import Iterable, List, Tuple

import networkx as nx
from pydantic import BaseModel


class TemporalChanges(BaseModel):
    deletions: List[Tuple]
    insertions: List[Tuple]


class TemporalGraph:
    def __init__(self, base_graph: nx.Graph, steps: List[TemporalChanges]):
        self.base_graph = base_graph
        self.steps = steps
        # cached_snapshots[i] is snapshot at time i
        self.cached_snapshots: List[nx.Graph] = [base_graph.copy()]

    def num_steps(self) -> int:
        return len(self.steps)

    def __len__(self) -> int:
        # number of snapshots including t=0
        return len(self.steps) + 1

    def __getitem__(self, idx: int) -> nx.Graph:
        """
        Return snapshot graph at time idx:
        - idx == 0           → base_graph
        - idx == k (1..T)    → base_graph with steps[0..k-1] applied
        Supports negative indices like Python lists.
        """
        if idx < 0:
            idx = self.__len__() + idx

        if not (0 <= idx <= len(self.steps)):
            raise IndexError(f"Snapshot index {idx} out of range [0, {len(self.steps)}]")

        # If we already have it, just return
        if idx < len(self.cached_snapshots):
            return self.cached_snapshots[idx]

        # Otherwise, build missing snapshots incrementally
        while len(self.cached_snapshots) <= idx:
            current_index = len(self.cached_snapshots)  # snapshot to create
            prev_graph = self.cached_snapshots[-1].copy()
            step = self.steps[current_index - 1]        # apply step for this snapshot
            prev_graph.remove_edges_from(step.deletions)
            prev_graph.add_edges_from(step.insertions)
            self.cached_snapshots.append(prev_graph)

        return self.cached_snapshots[idx]
    
    def iter_snapshots(self) -> Iterable[nx.Graph]:
        for i in range(len(self)):
            yield self[i]

if __name__ == "__main__":
    graph = nx.karate_club_graph()
    changes_step1 = TemporalChanges(
        deletions=[(0, 1), (2, 3)],
        insertions=[(0, 2), (1, 3)]
    )
    changes_step2 = TemporalChanges(deletions=[(4, 5)], insertions=[(5, 6)])
    temporal_graph = TemporalGraph(base_graph=graph, steps=[changes_step1, changes_step2])
    for snapshot in temporal_graph.iter_snapshots():
        print(snapshot)
