from typing import List, Dict, Set, Any
from cdlib import algorithms


class StaticMethods:
    NON_OVERLAP_METHODS: Set[str] = {
        algorithms.louvain.__name__,
        algorithms.infomap.__name__,
        algorithms.girvan_newman.__name__,
        algorithms.label_propagation.__name__,
        algorithms.fastgreedy.__name__,
        algorithms.walktrap.__name__,
    }
