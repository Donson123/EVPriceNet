import random
from typing import Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_station_graph(
    n_nodes: int = 20,
    area_size: tuple[float, float] = (1.0, 1.0),
    base_radius: float = 0.25,
    max_degree: int = 3,
    degree_probs: Optional[dict[int, float]] = None,
    competitor_share: float = 0.5,
    max_radius_factor: float = 2.0,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a NetworkX graph representing charging stations and their proximity links.

    Rules:
    - Each node has at least 1 edge and at most max_degree edges.
    - Edge count per node follows the given degree probability distribution.
    - 50% of nodes (by default) are competitors; the rest are owned stations.
    - Node attributes: 'is_competitor' (bool), 'pos' (tuple of coordinates).

    Returns:
        G: a NetworkX Graph representing the charge-station network.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if degree_probs is None:
        degree_probs = {1: 0.5, 2: 0.4, 3: 0.1}  # smaller chance of 3 edges

    # normalize degree probabilities
    allowed_degrees = [d for d in degree_probs.keys() if 1 <= d <= max_degree]
    probs = np.array([degree_probs[d] for d in allowed_degrees], dtype=float)
    probs = probs / probs.sum()

    # random 2D positions inside given area
    width, height = area_size
    positions = np.column_stack([
        rng.uniform(0, width, n_nodes),
        rng.uniform(0, height, n_nodes)
    ])

    # competitor share
    n_competitors_total = int(round(n_nodes * competitor_share))
    labels = ["ours"] * (n_nodes - n_competitors_total) + ["competitor"] * n_competitors_total
    rng.shuffle(labels)

    # initialize graph
    G = nx.Graph()
    for i in range(n_nodes):
        is_comp = labels[i] == "competitor"
        G.add_node(
            i,
            is_competitor=is_comp,
            pos=(float(positions[i, 0]), float(positions[i, 1])),
        )

    # pairwise distance matrix
    dists = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)

    # helper: return neighbors within radius
    def neighbors_within_radius(i, radius, exclude_connected=True):
        """Return nodes within distance 'radius' from node i."""
        cand = np.where(dists[i] <= radius)[0]
        cand = [j for j in cand if j != i]
        if exclude_connected:
            cand = [j for j in cand if not G.has_edge(i, j)]
        cand_sorted = sorted(cand, key=lambda j: dists[i, j])
        return cand_sorted

    # target degree per node, sampled from distribution
    target_deg = np.array(rng.choice(allowed_degrees, size=n_nodes, p=probs))
    target_deg = np.minimum(target_deg, max_degree)
    degrees = np.zeros(n_nodes, dtype=int)

    # connect nearby nodes up to target degree
    node_order = list(range(n_nodes))
    rng.shuffle(node_order)
    for i in node_order:
        if degrees[i] >= target_deg[i]:
            continue
        cand = neighbors_within_radius(i, base_radius)
        for j in cand:
            if degrees[i] >= target_deg[i]:
                break
            if degrees[i] < max_degree and degrees[j] < max_degree and not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(dists[i, j]))
                degrees[i] += 1
                degrees[j] += 1

    # ensure every node has at least one edge
    for i in range(n_nodes):
        if degrees[i] >= 1:
            continue
        radius = base_radius
        connected = False
        while radius <= base_radius * max_radius_factor and not connected:
            cand = neighbors_within_radius(i, radius)
            cand = [j for j in cand if degrees[j] < max_degree]
            if cand:
                j = cand[0]
                G.add_edge(i, j, weight=float(dists[i, j]))
                degrees[i] += 1
                degrees[j] += 1
                connected = True
                break
            radius *= 1.25  # slowly increase radius
        if not connected:
            # fallback: connect to closest node with capacity
            candidates = [j for j in range(n_nodes)
                          if j != i and degrees[j] < max_degree and not G.has_edge(i, j)]
            if candidates:
                j = min(candidates, key=lambda j2: dists[i, j2])
                G.add_edge(i, j, weight=float(dists[i, j]))
                degrees[i] += 1
                degrees[j] += 1

    # add extra edges up to target degree
    for i in node_order:
        while degrees[i] < target_deg[i] and degrees[i] < max_degree:
            cand = neighbors_within_radius(i, base_radius)
            cand = [j for j in cand if degrees[j] < max_degree]
            if not cand:
                break
            j = cand[0]
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(dists[i, j]))
                degrees[i] += 1
                degrees[j] += 1

    return G


def pick_focus_node(G: nx.Graph) -> int:
    """
    Pick a focus node (our station) that is not a competitor and has at least one edge.
    Falls back to any 'ours' node if none have edges, then to any node.
    """
    # prefer: our node with degree >= 1
    for n, data in G.nodes(data=True):
        if not data.get("is_competitor", False) and G.degree(n) >= 1:
            return n
    # fallback: any our node
    for n, data in G.nodes(data=True):
        if not data.get("is_competitor", False):
            return n
    # last fallback: any node
    return next(iter(G.nodes()))


def competitor_access_factor(G: nx.Graph, focus_node: int, scale_by_neighbors: bool = False) -> float:
    """
    Compute an access factor in [0,1] indicating whether a nearby competitor exists.
    - If scale_by_neighbors=False: 1.0 if at least one competitor neighbor exists else 0.0.
    - If True: scale by competitor-neighbor count (capped to 1.0).
    """
    if focus_node not in G:
        return 0.0
    comp_neighbors = sum(1 for nb in G.neighbors(focus_node)
                         if G.nodes[nb].get("is_competitor", False))
    if comp_neighbors <= 0:
        return 0.0
    if not scale_by_neighbors:
        return 1.0
    # cap by 3 competitor neighbors (consistent with your max_degree default)
    return min(1.0, comp_neighbors / 3.0)



# Example usage
if __name__ == "__main__":
    G = generate_station_graph(
        n_nodes=20,
        area_size=(2.0, 2.0),
        base_radius=0.35,
        max_degree=3,
        competitor_share=0.5,
        seed=42,
    )

    # visualization
    pos = nx.get_node_attributes(G, "pos")
    ours = [n for n, d in G.nodes(data=True) if not d["is_competitor"]]
    comps = [n for n, d in G.nodes(data=True) if d["is_competitor"]]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=ours, node_color="tab:blue", label="ours")
    nx.draw_networkx_nodes(G, pos, nodelist=comps, node_color="tab:orange", label="competitor")
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
