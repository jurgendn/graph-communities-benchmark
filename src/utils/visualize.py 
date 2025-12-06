import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# Run an overlapping community detector and visualize overlaps
sample_graph = nx.gaussian_random_partition_graph(
    n=40, s=15, v=2, p_in=0.7, p_out=0.1, directed=False
)

result = algorithms.angel(sample_graph, threshold=0.25)
communities = result.communities

# Calculate membership counts
membership_counts = {n: sum(n in c for c in communities) for n in sample_graph.nodes}
overlap_nodes = [n for n, cnt in membership_counts.items() if cnt > 1]

# Custom layout: circular arrangement of cluster centers
n_communities = len(communities)
pos = {}
radius_outer = 3.0  # radius for cluster centers
radius_inner = 0.6  # radius for nodes within each cluster

# Place cluster centers in a circle
cluster_centers = []
for i in range(n_communities):
    angle = 2 * np.pi * i / n_communities
    cx = radius_outer * np.cos(angle)
    cy = radius_outer * np.sin(angle)
    cluster_centers.append((cx, cy))

# Place nodes around their cluster centers
for node in sample_graph.nodes:
    if node in overlap_nodes:
        # Position overlapping nodes between their clusters
        member_idxs = [i for i, c in enumerate(communities) if node in c]
        if len(member_idxs) >= 2:
            # Place between first two communities
            cx1, cy1 = cluster_centers[member_idxs[0]]
            cx2, cy2 = cluster_centers[member_idxs[1]]
            # Midpoint with slight offset
            blend = 0.5 + 0.1 * (np.random.random() - 0.5)
            px = blend * cx1 + (1 - blend) * cx2
            py = blend * cy1 + (1 - blend) * cy2
            pos[node] = (px, py)
        else:
            # Fallback to first cluster
            cx, cy = cluster_centers[member_idxs[0]]
            pos[node] = (cx, cy)
    else:
        # Place non-overlapping nodes around their cluster center
        comm_idx = next((i for i, c in enumerate(communities) if node in c), None)
        if comm_idx is not None:
            cx, cy = cluster_centers[comm_idx]
            angle = 2 * np.pi * np.random.random()
            r = radius_inner * np.sqrt(np.random.random())
            pos[node] = (cx + r * np.cos(angle), cy + r * np.sin(angle))
        else:
            # Node not in any community - place at origin or skip
            pos[node] = (0, 0)

plt.figure(figsize=(10, 10))
ax = plt.gca()

# Draw edges
nx.draw_networkx_edges(sample_graph, pos, alpha=0.2, width=0.5)

cmap = plt.cm.get_cmap("tab20")
legend_handles = []

# Draw community nodes
for idx, comm in enumerate(communities):
    color = cmap(idx % cmap.N)
    non_overlap_comm = [n for n in comm if n not in overlap_nodes]
    if non_overlap_comm:
        nx.draw_networkx_nodes(
            sample_graph,
            pos,
            nodelist=non_overlap_comm,
            node_color=[color],
            node_size=150,
            alpha=0.7,
        )
    if idx < 8:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="w", label=f"Comm {idx + 1}",
                markerfacecolor=color, markersize=10, markeredgecolor="none"
            )
        )

# Draw overlapping nodes with distinct styling
if overlap_nodes:
    nx.draw_networkx_nodes(
        sample_graph, pos, nodelist=overlap_nodes,
        node_color="white", edgecolors="red", linewidths=3,
        node_size=300, alpha=1.0, zorder=10
    )
    nx.draw_networkx_labels(
        sample_graph, pos,
        labels={n: str(membership_counts[n]) for n in overlap_nodes},
        font_size=10, font_color="red", font_weight="bold"
    )
    legend_handles.append(
        Line2D(
            [0], [0], marker="o", color="w", label="Overlap node (#comms)",
            markerfacecolor="none", markeredgecolor="red", markersize=12, markeredgewidth=3
        )
    )

plt.axis("off")
plt.title("Overlapping communities - Circular layout with overlap nodes between clusters", fontsize=14)
plt.legend(handles=legend_handles, loc="upper left", frameon=False)
plt.tight_layout()
plt.show()
