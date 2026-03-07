import networkx as nx
from cdlib import evaluation

from src.evaluations.target_modularity import overlapping_modularity_q0


def get_evaluation_functions(
    overlap: bool = False,
):
    if overlap is True:
        return overlapping_modularity_q0
    else:
        return evaluation.newman_girvan_modularity
