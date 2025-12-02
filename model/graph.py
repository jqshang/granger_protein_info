import networkx as nx
from tqdm.auto import tqdm
from collections import deque


def dfs(DG, source, target):
    stack = [(source, [source])]

    while stack:
        (node, path) = stack.pop()

        for nbr in DG.successors(node):
            if nbr in path:
                continue
            new_path = path + [nbr]
            if nbr == target:
                yield new_path
            else:
                stack.append((nbr, new_path))


def filter_paths(DG, amino_acids, allosteric_sites, active_sites):
    name_to_idx = {name: i for i, name in enumerate(amino_acids)}

    allosteric_indices = {name: name_to_idx[name] for name in allosteric_sites}
    active_indices = {name: name_to_idx[name] for name in active_sites}

    paths_by_pair = {}
    used_nodes = set()
    used_edges = set()

    allo_items = list(allosteric_indices.items())
    active_items = list(active_indices.items())
    total_pairs = len(allo_items) * len(active_items)

    pbar_pairs = tqdm(total=total_pairs, desc="Allosteric-Active pairs")

    for allo_name, allo_idx in allo_items:
        for active_name, target_idx in active_items:
            pbar_pairs.update(1)

            paths_idx = list(dfs(DG, allo_idx, target_idx))

            paths_names = []
            for p in paths_idx:
                used_nodes.update(p)
                for u, v in zip(p, p[1:]):
                    used_edges.add((u, v))

                paths_names.append([amino_acids[i] for i in p])

            paths_by_pair[(allo_name, active_name)] = paths_names

    pbar_pairs.close()

    signaling_subgraph = DG.edge_subgraph(used_edges).copy()
    return paths_by_pair, signaling_subgraph


def _reachable_from_sources(DG, sources):
    visited = set(sources)
    dq = deque(sources)

    while dq:
        u = dq.popleft()
        for v in DG.successors(u):
            if v not in visited:
                visited.add(v)
                dq.append(v)

    return visited


def filter_paths_new(DG, amino_acids, allosteric_sites, active_sites):
    name_to_idx = {name: i for i, name in enumerate(amino_acids)}

    allo_idx = [name_to_idx[name] for name in allosteric_sites]
    active_idx = [name_to_idx[name] for name in active_sites]

    forward = _reachable_from_sources(DG, allo_idx)

    DG_rev = DG.reverse(copy=False)
    backward = _reachable_from_sources(DG_rev, active_idx)

    used_edges = set()
    for u, v in DG.edges():
        if (u in forward) and (v in backward):
            used_edges.add((u, v))

    signaling_subgraph = DG.edge_subgraph(used_edges).copy()
    used_nodes = set(signaling_subgraph.nodes())

    return signaling_subgraph, used_nodes, used_edges


def build_scc_quotient_graph(DG, amino_acids):
    sccs = list(nx.strongly_connected_components(DG))

    node_to_scc = {}
    for cid, comp in enumerate(sccs):
        for node in comp:
            node_to_scc[node] = cid

    Q = nx.DiGraph()

    for cid, comp in enumerate(sccs):
        members_idx = sorted(list(comp))
        members_names = [amino_acids[i] for i in members_idx]
        label = ",".join(members_names)

        Q.add_node(
            cid,
            members_idx=members_idx,
            members_names=members_names,
            label=label,
        )

    for u, v in DG.edges():
        cu = node_to_scc[u]
        cv = node_to_scc[v]
        if cu != cv:
            Q.add_edge(cu, cv)

    return sccs, node_to_scc, Q
