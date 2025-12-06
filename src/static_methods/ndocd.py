from typing import Any, List, Optional, Set
from cdlib import NodeClustering
import networkx as nx


class NDOCD:
    """
    Implementation of Overlapping Community Detection based on Network Decomposition (NDOCD).
    Ref: Ding et al. (2016)
    """
    def __init__(self, beta: float = 0.3, 
                 threshold_js: float = 0.4, threshold_md: float = 0.5, 
                 min_community_size: int = 3):
        """
        Initialize the NDOCD parameters. The graph is passed to the execute method.
        
        Args:
            beta: Parameter for CNFV (default 0.3).
            threshold_js: Threshold for Joint Strength (default 0.4).
            threshold_md: Threshold for Membership Degree (default 0.5).
            min_community_size: Minimum size to retain a community.
        """
        self.beta = beta
        self.threshold_js = threshold_js
        self.threshold_md = threshold_md
        self.min_community_size = min_community_size
        self.G_curr: Optional[nx.Graph] = None # Graph copy for decomposition

    def _calculate_cnfv(self, node: Any, N: int) -> float:
        """Calculates Comprehensive Network Feature Value (CNFV)."""
        if self.G_curr is None or self.G_curr.degree(node) == 0:
            return -1.0
        
        k_i = self.G_curr.degree(node)
        # Clustering coefficient must be calculated on the current, decomposed graph
        C_i = nx.clustering(self.G_curr, node)
        
        return (self.beta * C_i) + ((1 - self.beta) * (k_i / N))

    def _get_centered_clique(self) -> Optional[Set[Any]]:
        """Step 1: Seed Selection using CNFV."""
        if self.G_curr is None: return None
        
        nodes = list(self.G_curr.nodes())
        if not nodes:
            return None

        N = len(nodes)
        
        # 1. Select vertex x with highest CNFV
        try:
            best_node = max(nodes, key=lambda n: self._calculate_cnfv(n, N))
        except ValueError: # Happens if all nodes have CNFV -1 (i.e., isolated)
            return None
        
        if self.G_curr.degree(best_node) == 0:
            return None

        # 2. Build clique centered in x
        clique = {best_node}
        neighbors = list(self.G_curr.neighbors(best_node))
        neighbors.sort(key=lambda x: self.G_curr.degree(x), reverse=True)

        for neighbor in neighbors:
            if all(self.G_curr.has_edge(neighbor, member) for member in clique):
                clique.add(neighbor)

        return clique

    def _expand_seed(self, seed: Set[Any]) -> Set[Any]:
        """Step 2: Seed Expansion using Joint Strength (JS) and Membership Degree (MD)."""
        if self.G_curr is None: return seed
        
        community = set(seed)
        
        while True:
            added = False
            shell = set()
            for node in community:
                shell.update(n for n in self.G_curr.neighbors(node) if n not in community)
            
            n_k = len(community) 
            candidates = []

            for node in shell:
                m_ik = sum(1 for member in community if self.G_curr.has_edge(node, member))
                k_i = self.G_curr.degree(node)

                if k_i == 0:
                    continue

                js = m_ik / n_k
                md = m_ik / k_i
                
                if js > self.threshold_js or md > self.threshold_md:
                    candidates.append(node)
            
            for cand in candidates:
                if cand not in community:
                    community.add(cand)
                    added = True
            
            if not added:
                break
                
        return community

    def execute(self, G: nx.Graph) -> NodeClustering:
        """
        Executes the main NDOCD loop.
        
        Args:
            G: The input NetworkX graph.
            
        Returns:
            NodeClustering
        """
        # Initialize graph copy for decomposition
        self.G_curr = G.copy()
        final_communities: List[Set[Any]] = []
        
        # Loop until no edges remain
        while self.G_curr.number_of_edges() > 0:
            
            # 1. Seed Selection
            seed = self._get_centered_clique()
            
            # Break if no valid seed can be formed
            if not seed or len(seed) < 2:
                break
            
            # 2. Seed Expansion
            community = self._expand_seed(seed)
            
            # Filter: discard small communities
            if len(community) >= self.min_community_size:
                final_communities.append(community)
            
            # 3. Network Decomposition (Remove internal links)
            edges_to_remove = []
            comm_list = list(community)
            for i in range(len(comm_list)):
                for j in range(i + 1, len(comm_list)):
                    u, v = comm_list[i], comm_list[j]
                    if self.G_curr.has_edge(u, v):
                        edges_to_remove.append((u, v))
            
            if not edges_to_remove:
                # Should not happen if len(community) >= 2, but for safety
                break
                
            self.G_curr.remove_edges_from(edges_to_remove)

        # Clean up the instance variable
        self.G_curr = None
        
        # 4. Construct and return the NodeCluster
        # Convert sets to lists because cdlib expects indexable communities
        communities_as_lists = [list(comm) for comm in final_communities]

        node_clustering = NodeClustering(
            communities=communities_as_lists,
            graph=G,
            method_name="NDOCD",
            method_parameters={
                "beta": self.beta,
                "threshold_js": self.threshold_js,
                "threshold_md": self.threshold_md,
                "min_community_size": self.min_community_size,
            },
            overlap=True,
        )
        return node_clustering


    def __call__(self, G: nx.Graph) -> NodeClustering:
        return self.execute(G)