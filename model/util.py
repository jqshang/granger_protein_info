import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def coords_to_residue_scores(S, n_coords=3):
    """
    S: (D, D) GC strengths between *coordinates*.
       D = num_acids * n_coords

    Returns:
      S_res: (R, R) residue-level strengths, R = D // n_coords
             S_res[i, j] ~ influence of residue j -> residue i
    """
    D = S.shape[0]
    assert D % n_coords == 0, "D must be divisible by n_coords"
    R = D // n_coords

    # reshape to (target_res, coord_d, source_res, coord_j)
    S_reshaped = S.reshape(R, n_coords, R, n_coords)
    S_res = S_reshaped.mean(axis=(1, 3))  # average over coordinate dims
    return S_res


def build_residue_adjacency(S_res, percentile=95):
    """
    S_res: (R, R) residue-level scores
    percentile: global percentile threshold (e.g., 95 keeps top 5% edges)

    Returns:
      G_res: (R, R) binary adjacency, G_res[i, j] = 1 if j -> i
      tau:   threshold used
    """
    tau = np.percentile(S_res, percentile)
    G_res = (S_res > tau).astype(int)
    np.fill_diagonal(G_res, 0)
    return G_res, tau


def adjacency_to_digraph(G, node_names=None):
    """
    G: (N, N) numpy array, G[i, j] = 1 if j -> i (edge j -> i)
    node_names: optional list of labels (len N)

    Returns:
      DG: networkx.DiGraph
    """
    G = np.asarray(G)
    N = G.shape[0]
    DG = nx.DiGraph()

    # add nodes
    if node_names is None:
        DG.add_nodes_from(range(N))
    else:
        assert len(node_names) == N
        for idx, name in enumerate(node_names):
            DG.add_node(idx, label=name)

    # add edges (j -> i)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if G[i, j] != 0:
                DG.add_edge(j, i)

    return DG


def plot_digraph(DG, node_names=None, title=None, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(DG, k=0.3, iterations=200)

    if node_names is None:
        labels = {n: n for n in DG.nodes()}
    else:
        labels = {i: node_names[i] for i in DG.nodes()}

    nx.draw_networkx_nodes(DG, pos, node_size=300)
    nx.draw_networkx_edges(DG, pos, arrows=True, arrowstyle="->", arrowsize=12)
    nx.draw_networkx_labels(DG, pos, labels, font_size=8)

    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
