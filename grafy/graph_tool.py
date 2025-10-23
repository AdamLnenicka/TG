"""Simple .tg graph parser and CLI utilities.

Usage examples:
  python graph_tool.py summary 01.tg
  python graph_tool.py shortest 01.tg --src A --dst F

This script parses files with lines like:
  u A;
  h A > B 3 :h1;
  h A - B 5;
  h A < B 2;

Interpretation:
  - 'u <name>;' declares a vertex
  - 'h X > Y w :label;' directed edge X -> Y (weight optional)
  - 'h X < Y w :label;' directed edge Y -> X
  - 'h X - Y w ;' undirected edge

The script builds a networkx graph and exposes common analyses.
"""

from __future__ import annotations
import argparse
import re
from typing import List, Dict, Tuple, Optional
import networkx as nx
import sys

try:
    import networkx as nx  # try re-import for clarity
    USE_NX = True
except Exception:
    nx = None
    USE_NX = False


    # ... SimpleGraph removed - this CLI-only tool relies on networkx


NODE_RE = re.compile(r"^\s*u\s+([^\s\[;]+|\*)\s*(?:\[?(-?\d+(?:\.\d+)?)\]?)?\s*;\s*$")
EDGE_RE = re.compile(r"^\s*h\s+(.+?)\s*;\s*$")


def parse_tg(path: str) -> Dict:
    nodes: List[Tuple[str, Optional[float]]] = []
    edges: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = NODE_RE.match(line)
            if m:
                ident = m.group(1)
                # omitted node marker '*' -> skip
                if ident == "*":
                    continue
                weight = None
                w = m.group(2)
                if w:
                    try:
                        weight = float(w)
                    except ValueError:
                        weight = None
                nodes.append((ident, weight))
                continue
            m2 = EDGE_RE.match(line)
            if m2:
                body = m2.group(1)
                # robust parse: allow optional numeric weight and optional :label
                m3 = re.match(r"^\s*([^\s]+)\s*(<|>|-)\s*([^\s]+)(?:\s+(-?\d+(?:\.\d+)?))?(?:\s*:(\S+))?\s*$", body)
                if not m3:
                    continue
                a = m3.group(1)
                rel = m3.group(2)
                b = m3.group(3)
                weight: Optional[float] = None
                label: Optional[str] = None
                if m3.group(4):
                    try:
                        weight = float(m3.group(4))
                    except ValueError:
                        weight = None
                if m3.group(5):
                    label = m3.group(5)
                edges.append({"a": a, "b": b, "rel": rel, "weight": weight, "label": label, "raw": body})
                continue
            # ignore unknown lines
    return {"nodes": nodes, "edges": edges}


def build_graph(parsed: Dict):
    # If there are any directed edges (rel '>' or '<'), create DiGraph, otherwise Graph
    directed_present = any(e["rel"] in (">", "<") for e in parsed["edges"]) or False
    if directed_present:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # add nodes
    for n in parsed["nodes"]:
        if isinstance(n, tuple):
            ident, weight = n
            G.add_node(ident, weight=weight)
        else:
            G.add_node(n)

    for e in parsed["edges"]:
        a = e["a"]
        b = e["b"]
        rel = e["rel"]
        w = e.get("weight")
        label = e.get("label")
        if rel == "-":
            # undirected
            if isinstance(G, nx.DiGraph):
                # add both directions for analysis convenience
                G.add_edge(a, b, weight=w, label=label, undirected=True)
                G.add_edge(b, a, weight=w, label=label, undirected=True)
            else:
                G.add_edge(a, b, weight=w, label=label)
        elif rel == ">":
            G.add_edge(a, b, weight=w, label=label)
        elif rel == "<":
            # a < b means edge from b -> a
            G.add_edge(b, a, weight=w, label=label)
        else:
            # fallback: treat as undirected
            G.add_edge(a, b, weight=w, label=label)

    return G


def summarize(G) -> Dict:
    directed = G.is_directed()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    weights = [d.get("weight") for _, _, d in G.edges(data=True) if d.get("weight") is not None]
    weighted = len(weights) > 0
    loops_edges = [(u, v, d) for u, v, d in G.edges(data=True) if u == v]
    multiedge = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    # isolated nodes: degree 0
    isolated = [n for n, d in G.degree() if d == 0]
    # count duplicate edges if not a MultiGraph: look for parallel edges in edge list
    multi_count = 0
    try:
        # build a normalized unordered key for undirected, ordered for directed
        seen = {}
        seen_edges = {}
        for u, v, d in G.edges(data=True):
            key = (u, v) if G.is_directed() else tuple(sorted((u, v)))
            seen.setdefault(key, 0)
            seen[key] += 1
            seen_edges.setdefault(key, [])
            seen_edges[key].append((u, v, d))
        multi_count = sum(1 for k, c in seen.items() if c > 1)
        multi_details = [vals for k, vals in seen_edges.items() if len(vals) > 1]
    except Exception:
        multi_count = 0
        multi_details = []
    info = {
        "directed": directed,
        "nodes": n,
        "edges": m,
        "weighted": weighted,
        "loops": len(loops_edges),
        "loops_list": loops_edges,
        "multigraph": multiedge,
        "multi_edges_count": multi_count,
        "multi_edges_details": multi_details,
        "isolated_nodes": len(isolated),
        "isolated_nodes_list": isolated,
    }
    # degrees
    if directed:
        info["in_degrees"] = dict(G.in_degree())
        info["out_degrees"] = dict(G.out_degree())
        info["scc_count"] = nx.number_strongly_connected_components(G)
    else:
        info["degrees"] = dict(G.degree())
        info["components"] = nx.number_connected_components(G)

    # basic properties
    try:
        info["is_dag"] = nx.is_directed_acyclic_graph(G) if directed else None
    except Exception:
        info["is_dag"] = None

    return info


def print_summary(info: Dict):
    print("Shrnutí grafu:")
    print(f"  orientovaný: {info['directed']}")
    print(f"  počet uzlů: {info['nodes']}")
    print(f"  počet hran: {info['edges']}")
    print(f"  ohodnocený (váhy): {info['weighted']}")
    print(f"  smyčky: {info['loops']}")
    print(f"  multigraf: {info['multigraph']}")
    print(f"  počet násobných hran (paralelních): {info.get('multi_edges_count', 0)}")
    print(f"  počet izolovaných uzlů: {info.get('isolated_nodes', 0)}")
    if info['directed']:
        print(f"  počet silně souvisejících komponent: {info.get('scc_count')}")
        print("  výstupní stupně:")
        for k, v in sorted(info.get('out_degrees', {}).items(), key=lambda kv: str(kv[0])):
            print(f"    {k}: {v}")
    else:
        print(f"  počet souvisejících komponent: {info.get('components')}")
        print("  stupně uzlů:")
        for k, v in sorted(info.get('degrees', {}).items(), key=lambda kv: str(kv[0])):
            print(f"    {k}: {v}")


def cmd_summary(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    info = summarize(G)
    print_summary(info)


def cmd_shortest(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    src = args.src
    dst = args.dst
    if src not in G or dst not in G:
        print("Počáteční nebo cílový uzel není v grafu")
        return
    # if weights present, use Dijkstra, else BFS
    has_weights = any(d.get("weight") is not None for _, _, d in G.edges(data=True))
    try:
        if has_weights:
            path = nx.shortest_path(G, source=src, target=dst, weight="weight")
            length = nx.shortest_path_length(G, source=src, target=dst, weight="weight")
        else:
            path = nx.shortest_path(G, source=src, target=dst)
            length = len(path) - 1
        print(f"Nejkratší cesta {src} -> {dst}: délka={length}")
        print(" -> ".join(path))
    except nx.NetworkXNoPath:
        print("Cesta neexistuje")


def cmd_mst(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    if G.is_directed():
        print("MST má smysl pro neorientované grafy. Používám neorientovaný pohled.")
        U = G.to_undirected(as_view=False)
    else:
        U = G
    # ensure weights exist
    if not any(d.get("weight") is not None for _, _, d in U.edges(data=True)):
        print("Graf nemá váhy; MST bude počítána s jednotkovými vahami.")
    T = nx.minimum_spanning_tree(U, weight="weight")
    print("Hrany minimální kostry:")
    for u, v, d in T.edges(data=True):
        print(f"  {u} - {v} (váha={d.get('weight')})")


def cmd_centrality(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    k = args.top
    print("Stupňová centralita (top {}):".format(k))
    if G.is_directed():
        deg = dict(G.out_degree())
    else:
        deg = dict(G.degree())
    for node, val in sorted(deg.items(), key=lambda x: -x[1])[:k]:
        print(f"  {node}: {val}")
    try:
        btw = nx.betweenness_centrality(G)
        print("Betweenness centralita (top {}):".format(k))
        for node, val in sorted(btw.items(), key=lambda x: -x[1])[:k]:
            print(f"  {node}: {val:.4f}")
    except Exception as e:
        print("Výpočet betweenness selhal:", e)


def cmd_components(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    if G.is_directed():
        scc = list(nx.strongly_connected_components(G))
        print(f"Silně související komponenty: {len(scc)}")
        for i, comp in enumerate(scc, 1):
            print(f"  komponenta {i}: {sorted(comp)}")
    else:
        cc = list(nx.connected_components(G))
        print(f"Související komponenty: {len(cc)}")
        for i, comp in enumerate(cc, 1):
            print(f"  komponenta {i}: {sorted(comp)}")


def cmd_cycles(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    if not G.is_directed():
        cycles = list(nx.cycle_basis(G))
        if cycles:
            print(f"Nalezené cykly (počet={len(cycles)}):")
            for c in cycles:
                print("  ", c)
        else:
            print("Žádné cykly nebyly nalezeny")
    else:
        try:
            cycle = nx.find_cycle(G, orientation='original')
            print("Nalezen orientovaný cyklus:")
            print(cycle)
        except nx.NetworkXNoCycle:
            print("Nebyl nalezen žádný orientovaný cyklus")


def cmd_topo(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    if not G.is_directed():
        print("Topologické řazení je možné pouze pro orientované acyklické grafy (DAG)")
        return
    try:
        order = list(nx.topological_sort(G))
        print("Topologické řazení:")
        print(" -> ".join(order))
    except Exception as e:
        print("Topologické řazení selhalo (graf může mít cykly):", e)


def cmd_batch(args):
    import glob, json, os
    files = sorted(glob.glob(os.path.join(args.dir, "*.tg")))
    report = {}
    for f in files:
        parsed = parse_tg(f)
        G = build_graph(parsed)
        info = summarize(G)
        report[os.path.basename(f)] = info
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        print(f"Uložil jsem dávkový report do {args.out}")
    else:
        print(json.dumps(report, indent=2))


# Glossary and define command removed — this tool is CLI-only and minimal


def is_weighted(G):
    return any(d.get('weight') is not None for _, _, d in G.edges(data=True))


def is_oriented(G):
    return G.is_directed()


def is_connected(G):
    try:
        if G.is_directed():
            return nx.is_weakly_connected(G)
        else:
            return nx.is_connected(G)
    except Exception:
        if hasattr(G, 'connected_components'):
            return len(G.connected_components()) == 1
        return False


def is_simple(G):
    try:
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            return False
    except Exception:
        pass
    for u, v, d in G.edges(data=True):
        if u == v:
            return False
    return True


def is_multigraph(G):
    try:
        return isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    except Exception:
        return False


def is_planar(G):
    try:
        return nx.check_planarity(G, False)[0]
    except Exception:
        return False


def is_finite(G):
    return G.number_of_nodes() < float('inf')


def is_complete(G):
    try:
        return nx.is_connected(G) and nx.density(G) == 1.0
    except Exception:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        return m == n*(n-1)/2


def is_regular(G):
    degs = [d for _, d in G.degree()]
    return len(set(degs)) == 1


def is_bipartite(G):
    try:
        return nx.is_bipartite(G)
    except Exception:
        return False


def node_successors(G, node):
    if G.is_directed():
        try:
            return list(G.successors(node))
        except Exception:
            return [v for u, v, _ in G.edges(data=True) if u == node]
    else:
        return list(G.adj.get(node, {}).keys())


def node_predecessors(G, node):
    if G.is_directed():
        try:
            return list(G.predecessors(node))
        except Exception:
            return [u for u, v, _ in G.edges(data=True) if v == node]
    else:
        return list(G.adj.get(node, {}).keys())


def node_neighbors(G, node):
    # Neighbors should include all nodes connected by an edge (both in- and out-neighbors for directed graphs)
    try:
        if G.is_directed():
            # union of predecessors and successors
            succ = set(G.successors(node)) if hasattr(G, 'successors') else set(v for u, v, _ in G.edges(data=True) if u == node)
            pred = set(G.predecessors(node)) if hasattr(G, 'predecessors') else set(u for u, v, _ in G.edges(data=True) if v == node)
            return sorted(list(succ | pred))
        else:
            return sorted(list(G.neighbors(node)))
    except Exception:
        # fallback to adjacency dict if present
        try:
            neighbors = set(G.adj.get(node, {}).keys())
            # include incoming for directed structures stored in adj
            for u, targets in getattr(G, 'adj', {}).items():
                if node in targets:
                    neighbors.add(u)
            return sorted(list(neighbors))
        except Exception:
            return []


def out_neighborhood(G, node):
    return node_successors(G, node)


def in_neighborhood(G, node):
    return node_predecessors(G, node)


def neighborhood(G, node):
    s = set(node_neighbors(G, node))
    return sorted(list(s))


def edge_neighbors(G, u, v):
    # edges that share an endpoint with (u,v), excluding the edge itself
    neigh = []
    for a, b, d in G.edges(data=True):
        if (a == u and b == v) or (a == v and b == u):
            continue
        if a == u or b == u or a == v or b == v:
            neigh.append((a, b, d))
    return neigh


def clear_screen():
    import os
    try:
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    except Exception:
        pass


def out_degree_node(G, node):
    try:
        return G.out_degree(node) if G.is_directed() else G.degree(node)
    except Exception:
        return 0


def in_degree_node(G, node):
    try:
        return G.in_degree(node) if G.is_directed() else G.degree(node)
    except Exception:
        return 0


def degree_node(G, node):
    try:
        return G.degree(node)
    except Exception:
        return 0


def adjacency_matrix_str(G):
    nodes, mat = matrix_adjacency(G)
    out = "Matice sousednosti (řádky/sloupce = {})\n".format(nodes)
    for i, row in enumerate(mat):
        out += f"{nodes[i]} {row}\n"
    return out


def incidence_matrix_str(G):
    nodes, edges, mat = matrix_incidence(G)
    out = "Matice incidence (uzly x hrany):\n"
    for i, row in enumerate(mat):
        out += f"{nodes[i]} {row}\n"
    return out


def count_in_matrix(mat, nodes=None, edges=None, target=None):
    # target may be numeric or the string 'inf'
    cnt = 0
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if target == 'inf':
                try:
                    import math
                    if val == math.inf:
                        cnt += 1
                except Exception:
                    pass
            else:
                try:
                    if float(val) == float(target):
                        cnt += 1
                except Exception:
                    # if mat contains non-numeric like 0/1 as ints, compare directly
                    if val == target:
                        cnt += 1
    return cnt


def row_counts(mat):
    # return list of counts of non-zero entries per row
    res = []
    for row in mat:
        c = 0
        for v in row:
            try:
                if v != 0 and v is not None:
                    c += 1
            except Exception:
                if v:
                    c += 1
        res.append(c)
    return res


def col_counts(mat):
    if not mat:
        return []
    ncols = len(mat[0])
    res = [0]*ncols
    for row in mat:
        for j, v in enumerate(row):
            try:
                if v != 0 and v is not None:
                    res[j] += 1
            except Exception:
                if v:
                    res[j] += 1
    return res


def export_matrix_to_file(mat, row_names=None, col_names=None, path='matrix.txt'):
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            if row_names:
                # write header with column names
                if col_names:
                    fh.write('\t' + '\t'.join(map(str, col_names)) + '\n')
            for i, row in enumerate(mat):
                prefix = f"{row_names[i]}\t" if row_names else ''
                fh.write(prefix + '\t'.join(str(x) for x in row) + '\n')
        return True, path
    except Exception as e:
        return False, str(e)


def distance_matrix_str(G):
    try:
        import math
        nodes = sorted(list(G.nodes()))
        n = len(nodes)
        dmat = [[math.inf]*n for _ in range(n)]
        for i in range(n):
            dmat[i][i] = 0
        for u, v, meta in G.edges(data=True):
            i = nodes.index(u)
            j = nodes.index(v)
            w = meta.get('weight') if meta and meta.get('weight') is not None else 1
            dmat[i][j] = w
            if not G.is_directed():
                dmat[j][i] = w
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dmat[i][k] + dmat[k][j] < dmat[i][j]:
                        dmat[i][j] = dmat[i][k] + dmat[k][j]
        out = "Matice délek (uzly={})\n".format(nodes)
        for i in range(n):
            out += f"{nodes[i]} {dmat[i]}\n"
        return out
    except Exception:
        return "Nelze vypočítat matici délek"


def signed_matrix_str(G):
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    mat = [[0]*n for _ in range(n)]
    for u, v, d in G.edges(data=True):
        i = nodes.index(u)
        j = nodes.index(v)
        mat[i][j] = 1 if not G.is_directed() else 1
        if G.is_directed():
            mat[j][i] = -1
    out = "Znaménková matice (řádky/sloupce = {})\n".format(nodes)
    for i, row in enumerate(mat):
        out += f"{nodes[i]} {row}\n"
    return out


def incident_edges_table_str(G):
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append((f"{u}-{v}", u, v, d.get('weight'), d.get('label')))
    out = "Tabulka incidentních hran (hrana, u, v, váha, label):\n"
    for r in rows:
        out += f"{r}\n"
    return out


def adjacency_list_str(G):
    out = "Dynamický seznam sousedů (adjacency list):\n"
    for n in sorted(list(G.nodes())):
        try:
            neigh = list(G.neighbors(n))
        except Exception:
            neigh = list(G.adj.get(n, {}).keys())
        out += f"{n}: {neigh}\n"
    return out


def nodes_edges_list_str(G):
    out = "Dynamický seznam uzlů a hran:\n"
    out += f"Uzly: {list(G.nodes())}\n"
    out += "Hrany:\n"
    for u, v, d in G.edges(data=True):
        out += f"  {u} - {v} (váha={d.get('weight')}, label={d.get('label')})\n"
    return out


def cmd_gui(args):
    try:
        import tkinter as tk
        from tkinter import scrolledtext, filedialog, messagebox
    except Exception:
        print("Tkinter není dostupný v tomto Pythonu. Nelze otevřít GUI.")
        return

    root = tk.Tk()
    root.title("Graph Tool — GUI")
    root.geometry("900x600")

    # left frame for controls
    left = tk.Frame(root, width=300)
    left.pack(side='left', fill='y')
    # right frame for output
    right = tk.Frame(root)
    right.pack(side='right', expand=True, fill='both')

    txt = scrolledtext.ScrolledText(right)
    txt.pack(expand=True, fill='both')

    selected_graph = {'path': None, 'G': None}

    def load_file():
        p = filedialog.askopenfilename(filetypes=[('TG files', '*.tg'), ('All', '*.*')])
        if not p:
            return
        selected_graph['path'] = p
        parsed = parse_tg(p)
        selected_graph['G'] = build_graph(parsed)
        txt.delete('1.0', 'end')
        txt.insert('end', f"Načten soubor: {p}\n")

    def show_properties():
        G = selected_graph['G']
        if not G:
            messagebox.showinfo('Chyba', 'Nejprve načtěte graf')
            return
        txt.delete('1.0', 'end')
        out = []
        out.append(f"Ohodnocený (má hrany s vahami): {is_weighted(G)} — pokud některá hrana obsahuje 'váha', považujeme graf za ohodnocený.")
        out.append(f"Orientovaný: {is_oriented(G)} — každá hrana má směr (True) nebo ne (False).")
        out.append(f"Souvislý: {is_connected(G)} — zda existuje cesta mezi libovolnými dvěma uzly (u orientovaných grafů slabe/silně záleží na definici).")
        out.append(f"Prostý (bez smyček/násobných hran): {is_simple(G)} — True pokud nejsou smyčky ani duplicity hran.)")
        out.append(f"Jednoduchý (není multigraf): {not is_multigraph(G)} — True pokud graf není multigraf.")
        out.append(f"Rovinný: {is_planar(G)} — kontrola pomocí networkx (pokud dostupné).")
        out.append(f"Konečný: {is_finite(G)} — True pokud má konečný počet uzlů.")
        out.append(f"Úplný: {is_complete(G)} — True pokud je mezi každými dvěma uzly hrana.")
        out.append(f"Regulární: {is_regular(G)} — True pokud mají všechny uzly stejný stupeň.")
        out.append(f"Bipartitní: {is_bipartite(G)} — True pokud lze uzly rozdělit do dvou množin bez hran uvnitř množin.")
        txt.insert('end', '\n'.join(out) + '\n')

    # Inline representation chooser (no new window)
    rep_var = tk.StringVar(value='adj')
    tk.Label(left, text='Reprezentace:').pack()
    rep_opts = [('Matice sousednosti', 'adj'), ('Matice incidence', 'inc'), ('Matice délek', 'dist'), ('Znaménková matice', 'sign'), ('Tabulka incidentních hran', 'table'), ('Dynamický seznam sousedů', 'alist'), ('Uzly a hrany', 'nelist')]
    for (txtlab, val) in rep_opts:
        tk.Radiobutton(left, text=txtlab, variable=rep_var, value=val).pack(anchor='w')

    def show_rep_inline():
        G = selected_graph['G']
        if not G:
            messagebox.showinfo('Chyba', 'Nejprve načtěte graf')
            return
        txt.delete('1.0', 'end')
        choice = rep_var.get()
        if choice == 'adj':
            txt.insert('end', adjacency_matrix_str(G) + '\n')
        elif choice == 'inc':
            txt.insert('end', incidence_matrix_str(G) + '\n')
        elif choice == 'dist':
            txt.insert('end', distance_matrix_str(G) + '\n')
        elif choice == 'sign':
            txt.insert('end', signed_matrix_str(G) + '\n')
        elif choice == 'table':
            txt.insert('end', incident_edges_table_str(G) + '\n')
        elif choice == 'alist':
            txt.insert('end', adjacency_list_str(G) + '\n')
        elif choice == 'nelist':
            txt.insert('end', nodes_edges_list_str(G) + '\n')

    def node_query():
        G = selected_graph['G']
        if not G:
            messagebox.showinfo('Chyba', 'Nejprve načtěte graf')
            return
        n = node_entry.get().strip()
        if not n:
            messagebox.showinfo('Chyba', 'Zadejte jméno uzlu')
            return
        out = []
        out.append(f"Následníci: {node_successors(G, n)}")
        out.append(f"Předchůdci: {node_predecessors(G, n)}")
        out.append(f"Sousedé: {node_neighbors(G, n)}")
        out.append(f"Výstupní okolí: {out_neighborhood(G, n)}")
        out.append(f"Vstupní okolí: {in_neighborhood(G, n)}")
        out.append(f"Okolí: {neighborhood(G, n)}")
        out.append(f"Výstupní stupeň: {out_degree_node(G, n)}")
        out.append(f"Vstupní stupeň: {in_degree_node(G, n)}")
        out.append(f"Stupeň: {degree_node(G, n)}")
        txt.insert('end', '\n'.join(out) + '\n')

    tk.Button(left, text='Načíst .tg soubor', command=load_file).pack(fill='x')
    tk.Button(left, text='Vlastnosti grafu', command=show_properties).pack(fill='x')
    tk.Button(left, text='Zobrazit reprezentaci', command=show_rep_inline).pack(fill='x')
    tk.Label(left, text='Dotaz na uzel:').pack()
    node_entry = tk.Entry(left)
    node_entry.pack(fill='x')
    tk.Button(left, text='Zobrazit info o uzlu', command=node_query).pack(fill='x')
    tk.Button(left, text='Vymazat výstup', command=lambda: txt.delete('1.0', 'end')).pack(fill='x')
    tk.Button(left, text='Ukončit', command=root.destroy).pack(side='bottom', fill='x')

    root.mainloop()


def matrix_adjacency(G):
    nodes = sorted(list(G.nodes()))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # adjacency matrix should contain 1 where an edge exists (regardless of weight)
    mat = [[0]*n for _ in range(n)]
    for u, v, d in G.edges(data=True):
        i = idx[u]
        j = idx[v]
        mat[i][j] = 1
    return nodes, mat


def matrix_incidence(G):
    nodes = sorted(list(G.nodes()))
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append((u, v, d))
    m = len(edges)
    n = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    mat = [[0]*m for _ in range(n)]
    for j, (u, v, d) in enumerate(edges):
        i_u = idx[u]
        i_v = idx[v]
        mat[i_u][j] = 1
        mat[i_v][j] = 1
    return nodes, edges, mat


def cmd_analyze(args):
    parsed = parse_tg(args.file)
    G = build_graph(parsed)
    print("Analyza grafu:")
    print_summary(summarize(G))
    # adjacency
    nodes, adj = matrix_adjacency(G)
    print("\nMatice sousednosti (řádky/ sloupce = {}):".format(nodes))
    for i, row in enumerate(adj):
        print(nodes[i], row)
    # incidence
    nodes2, edges, inc = matrix_incidence(G)
    print("\nMatice incidence (uzly x hrany):")
    for i, row in enumerate(inc):
        print(nodes2[i], row)
    # degree lists
    if G.is_directed():
        print("\nVýstupní stupeň:", dict(G.out_degree()))
        print("Vstupní stupeň:", dict(G.in_degree()))
    else:
        print("\nStupně uzlů:", dict(G.degree()))
    # neighbors for each node
    print("\nSousedé (adj list):")
    for n in sorted(list(G.nodes())):
        if hasattr(G, 'adj'):
            # networkx Graph uses G[n]
            try:
                neigh = list(G[n].keys())
            except Exception:
                neigh = list(G.adj[n].keys())
        else:
            neigh = list(G.adj.get(n, {}).keys())
        print(f"  {n}: {neigh}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Jednoduchý nástroj pro práci se soubory .tg (grafy)")
    sub = parser.add_subparsers(dest="cmd")

    p_summary = sub.add_parser("summary", help="Zobrazit souhrn grafu")
    p_summary.add_argument("file", help="Cesta k .tg souboru")

    p_shortest = sub.add_parser("shortest", help="Najít nejkratší cestu mezi dvěma uzly")
    p_shortest.add_argument("file", help="Cesta k .tg souboru")
    p_shortest.add_argument("--src", required=True, help="Počáteční uzel")
    p_shortest.add_argument("--dst", required=True, help="Cílový uzel")

    p_mst = sub.add_parser("mst", help="Vypočítat minimální kostru (pro neorientovaný graf)")
    p_mst.add_argument("file", help="Cesta k .tg souboru")

    p_cent = sub.add_parser("centrality", help="Vypočítat centrality (top K)")
    p_cent.add_argument("file", help="Cesta k .tg souboru")
    p_cent.add_argument("--top", type=int, default=5, help="Kolik nejlepších uzlů vypsat")

    p_comp = sub.add_parser("components", help="Zobrazit komponenty grafu")
    p_comp.add_argument("file", help="Cesta k .tg souboru")

    p_cycle = sub.add_parser("cycles", help="Detekovat cykly v grafu")
    p_cycle.add_argument("file", help="Cesta k .tg souboru")

    p_topo = sub.add_parser("topo", help="Topologické řazení (pro DAG)")
    p_topo.add_argument("file", help="Cesta k .tg souboru")

    p_batch = sub.add_parser("batch", help="Dávkové vytvoření shrnutí pro všechny .tg v adresáři")
    p_batch.add_argument("--dir", default='.', help="Adresář s .tg soubory")
    p_batch.add_argument("--out", default=None, help="Soubor pro uložení JSON reportu")

    p_define = sub.add_parser("define", help="Zobrazit definici zadaného pojmu (česky)")
    p_define.add_argument("term", nargs="+", help="Heslo(k) ke zobrazení definice (uveďte v uvozovkách pro víceslovné)")

    p_analyze = sub.add_parser("analyze", help="Komplexní analýza grafu (matice, stupně, seznamy)")
    p_analyze.add_argument("file", help="Cesta k .tg souboru")

    args = parser.parse_args(argv)
    # interactive menu if no subcommand provided
    if args.cmd is None:
        # Simplified interactive CLI as requested: no GUI, only menu-driven CLI
        import glob, os

        def is_discrete(G):
            return G.number_of_edges() == 0

        def is_binary_tree(G):
            # Assumptions:
            # - For directed graphs: treat as rooted binary tree if exactly one node has in-degree 0,
            #   all others have in-degree 1, graph is weakly connected, acyclic and each node has out-degree <=2.
            # - For undirected graphs: treat as (unrooted) binary tree if it's a tree (connected, edges=n-1)
            #   and maximum degree <=3.
            n = G.number_of_nodes()
            m = G.number_of_edges()
            if n == 0:
                return False
            try:
                if G.is_directed():
                    # weakly connected
                    if not nx.is_weakly_connected(G):
                        return False
                    # acyclic
                    if not nx.is_directed_acyclic_graph(G):
                        return False
                    indeg = dict(G.in_degree())
                    zeros = [v for v, d in indeg.items() if d == 0]
                    if len(zeros) != 1:
                        return False
                    # everyone else has in-degree 1
                    if any(d > 1 for v, d in indeg.items() if v not in zeros):
                        return False
                    # out-degree <= 2
                    if any(d > 2 for _, d in G.out_degree()):
                        return False
                    return True
                else:
                    # undirected: check tree and max degree <=3
                    if not nx.is_connected(G):
                        return False
                    if m != n - 1:
                        return False
                    if max(d for _, d in G.degree()) > 3:
                        return False
                    return True
            except Exception:
                return False

        tg_files = sorted(glob.glob(os.path.join('.', '*.tg')))
        current_file = None
        # helper: clear screen if user enters a numeric response (int or float)
        def get_input(prompt: str) -> str:
            s = input(prompt).strip()
            if s:
                # clear if s represents an integer or float
                try:
                    float(s)
                    clear_screen()
                except Exception:
                    # not numeric -> do not clear
                    pass
            return s
        if tg_files:
            print("Nalezené .tg soubory v aktuálním adresáři:")
            for i, f in enumerate(tg_files, 1):
                print(f"  {i}. {os.path.basename(f)}")
            sel = get_input("Vyberte číslo souboru pro práci (nebo stiskněte Enter pro ruční zadání): ")
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(tg_files):
                    current_file = tg_files[idx]
                    print(f"Vybrán soubor: {os.path.basename(current_file)}")

        # load graph helper
        def load_graph(path):
            parsed = parse_tg(path)
            G = build_graph(parsed)
            return G

        G = None

        while True:
            print("\nHlavní nabídka:")
            print("  1. Vlastnosti grafu")
            print("  2. Reprezentace grafu")
            print("  3. Analýza uzlu")
            print("  4. Analýza hrany")
            print("  5. Operace nad grafem")
            print("  6. Změnit soubor")
            print("  7. Ukončit")
            choice = get_input("Volba: ")
            if choice == '7' or choice.lower() in ('q', 'exit'):
                return
            if choice == '6':
                # change file
                tg_files = sorted(glob.glob(os.path.join('.', '*.tg')))
                for i, f in enumerate(tg_files, 1):
                    print(f"  {i}. {os.path.basename(f)}")
                sel2 = get_input("Vyberte číslo souboru nebo zadejte cestu: ")
                if sel2.isdigit():
                    idx2 = int(sel2) - 1
                    if 0 <= idx2 < len(tg_files):
                        current_file = tg_files[idx2]
                        print(f"Vybrán soubor: {os.path.basename(current_file)}")
                elif sel2:
                    current_file = sel2
                    print(f"Vybrán soubor: {current_file}")
                else:
                    print("Soubor nezměněn")
                G = load_graph(current_file) if current_file else None
                continue
            # ensure graph loaded
            if not current_file:
                fname = input("Cesta k .tg souboru: ").strip()
                if not fname:
                    print("Nebyl zadán žádný soubor")
                    continue
                current_file = fname
            if G is None:
                try:
                    G = load_graph(current_file)
                except Exception as e:
                    print("Chyba při načítání grafu:", e)
                    current_file = None
                    G = None
                    continue

            # 1. Vlastnosti grafu
            if choice == '1':
                print("Vyberte vlastnost:")
                props = [
                    ("1", "Uzel grafu"),
                    ("2", "Hrany grafu"),
                    ("3", "Ohodnocený"),
                    ("4", "Orientovaný"),
                    ("5", "Souvislý"),
                    ("6", "Prostý"),
                    ("7", "Jednoduchý"),
                    ("8", "Multigraf"),
                    ("9", "Rovinný"),
                    ("10", "Konečný"),
                    ("11", "Úplný"),
                    ("12", "Regulární"),
                    ("13", "Bipartitní"),
                    ("14", "Diskrétní"),
                    ("15", "Binární strom"),
                    ("16", "Vypsat vše"),
                ]
                for k, v in props:
                    print(f"  {k}. {v}")
                sc = get_input("Volba: ")
                if sc == '1':
                    clear_screen()
                    print(list(G.nodes()))
                elif sc == '2':
                    clear_screen()
                    for i, (u, v, d) in enumerate(G.edges(data=True), 1):
                        print(f"  {i}. {u} - {v} (váha={d.get('weight')}, label={d.get('label')})")
                elif sc == '3':
                    print(is_weighted(G))
                elif sc == '4':
                    print(is_oriented(G))
                elif sc == '5':
                    print(is_connected(G))
                elif sc == '6':
                    print(is_simple(G))
                elif sc == '7':
                    print(not is_multigraph(G))
                elif sc == '8':
                    print(is_multigraph(G))
                elif sc == '9':
                    print(is_planar(G))
                elif sc == '10':
                    print(is_finite(G))
                elif sc == '11':
                    print(is_complete(G))
                elif sc == '12':
                    print(is_regular(G))
                elif sc == '13':
                    print(is_bipartite(G))
                elif sc == '14':
                    print(is_discrete(G))
                elif sc == '15':
                    print(is_binary_tree(G))
                elif sc == '16':
                    # vypsat vše: odpovědi na všechny stanovené možnosti
                    clear_screen()
                    print("Uzly grafu:")
                    print(list(G.nodes()))
                    print("\nHrany grafu:")
                    for i, (u, v, d) in enumerate(G.edges(data=True), 1):
                        print(f"  {i}. {u} - {v} (váha={d.get('weight')}, label={d.get('label')})")
                    print("\nOhodnocený:", is_weighted(G))
                    print("Orientovaný:", is_oriented(G))
                    print("Souvislý:", is_connected(G))
                    print("Prostý:", is_simple(G))
                    print("Jednoduchý:", not is_multigraph(G))
                    print("Multigraf:", is_multigraph(G))
                    info_all = summarize(G)
                    print("Multigraf:", is_multigraph(G))
                    print("Počet násobných hran (paralelních):", info_all.get('multi_edges_count'))
                    if info_all.get('multi_edges_count', 0) > 0:
                        print("  Detaily násobných hran (skupiny):")
                        for grp in info_all.get('multi_edges_details', []):
                            print("   - skupina:")
                            for u, v, d in grp:
                                print(f"      {u} - {v} (váha={d.get('weight')}, label={d.get('label')})")
                    print("Počet izolovaných uzlů:", info_all.get('isolated_nodes'))
                    if info_all.get('isolated_nodes', 0) > 0:
                        print("  Izolované uzly:", info_all.get('isolated_nodes_list'))
                    print("Smyčky (počet):", info_all.get('loops'))
                    if info_all.get('loops', 0) > 0:
                        print("  Smyčky:")
                        for u, v, d in info_all.get('loops_list', []):
                            print(f"    {u} - {v} (váha={d.get('weight')}, label={d.get('label')})")
                    print("Rovinný:", is_planar(G))
                    print("Konečný:", is_finite(G))
                    print("Úplný:", is_complete(G))
                    print("Regulární:", is_regular(G))
                    print("Bipartitní:", is_bipartite(G))
                    print("Diskrétní:", is_discrete(G))
                    print("Binární strom:", is_binary_tree(G))
                else:
                    print("Neplatná volba")

            # 2. Reprezentace grafu
            elif choice == '2':
                print("Reprezentace (má tři volby: 1) hodnota na pozici, 2) počet hodnot v matici, 3) zpět)")
                print("  1. Matice sousednosti (možnost mocniny)")
                print("  2. Matice incidence")
                print("  3. Matice délek")
                print("  4. Matice předchůdců")
                print("  5. Tabulka incidentních hran")
                rc = get_input("Volba (1-5): ")
                if rc not in ('1', '2', '3', '4', '5'):
                    print("Neplatná volba")
                    continue
                # --- adjacency matrix ---
                if rc == '1':
                    nodes, adj = matrix_adjacency(G)
                    k = get_input("Zadejte exponent k (celé >=1, výchozí 1): ")
                    try:
                        k = int(k) if k else 1
                    except Exception:
                        k = 1
                    n = len(nodes)
                    A = [[int(adj[i][j]) for j in range(n)] for i in range(n)]
                    def mat_mult(X, Y):
                        return [[sum(X[i][t] * Y[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
                    def mat_pow(M, p):
                        R = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
                        T = M
                        while p > 0:
                            if p & 1:
                                R = mat_mult(R, T)
                            T = mat_mult(T, T)
                            p >>= 1
                        return R
                    Ak = mat_pow(A, k)
                    print("  1. Hodnota na pozici (A B)")
                    print("  2. Počet hodnot v matici")
                    print("  3. Počet hodnot v řádku (non-zero count)")
                    print("  4. Počet hodnot ve sloupci (non-zero count)")
                    print("  5. Exportovat matici do textového souboru")
                    print("  6. Zpět")
                    sub = get_input("Volba: ")
                    if sub == '1':
                        pos = input("Zadejte uzel zdroj a cílový uzel (např. A B): ").strip()
                        if pos:
                            a, b = pos.split()[:2]
                            if a in nodes and b in nodes:
                                i = nodes.index(a); j = nodes.index(b)
                                clear_screen(); print(f"Hodnota A^{k}[{a},{b}] = {Ak[i][j]}")
                            else:
                                print("Neznámé uzly")
                    elif sub == '2':
                        cntq = input("Zadejte hodnotu pro spočítání výskytu v A^k (číslo nebo 'inf'): ").strip()
                        if cntq:
                            try:
                                target = float(cntq) if cntq != 'inf' else 'inf'
                            except Exception:
                                target = cntq
                            c = count_in_matrix(Ak, nodes=nodes, target=target)
                            clear_screen(); print(f"Počet hodnot {cntq} v A^{k}: {c}")
                    elif sub == '3':
                        rc = row_counts(Ak)
                        clear_screen(); print(f"Počet nenulových v každém řádku: {dict(zip(nodes, rc))}")
                    elif sub == '4':
                        cc = col_counts(Ak)
                        clear_screen(); print(f"Počet nenulových v každém sloupci: {dict(zip(nodes, cc))}")
                    elif sub == '5':
                        path = input("Cesta k výstupnímu souboru: ").strip() or 'adjacency_matrix.txt'
                        ok, msg = export_matrix_to_file(Ak, row_names=nodes, col_names=nodes, path=path)
                        if ok:
                            print(f"Vystup ulozen do: {msg}")
                        else:
                            print(f"Chyba při exportu: {msg}")
                    else:
                        continue
                # --- incidence matrix ---
                elif rc == '2':
                    nodes2, edges, inc = matrix_incidence(G)
                    print("  1. Hodnota na pozici")
                    print("  2. Počet hodnot")
                    print("  3. Počet hodnot v řádku")
                    print("  4. Počet hodnot ve sloupci")
                    print("  5. Exportovat matici do textového souboru")
                    print("  6. Zpět")
                    sub = get_input("Volba: ")
                    if sub == '1':
                        q = input("Zadejte uzel a (index hrany nebo label) — např. 'A 0' nebo 'A :e1' nebo 'A e1': ").strip()
                        if not q:
                            continue
                        parts = q.split()
                        if len(parts) < 2:
                            print("Neplatný vstup")
                            continue
                        u = parts[0]
                        key = parts[1]
                        # resolve edge by index or by label (allow label prefixed by ':' optionally)
                        if key.startswith(':'):
                            key = key[1:]
                        edge_idx = None
                        # try int index
                        try:
                            edge_idx = int(key)
                        except Exception:
                            # try to match by label
                            for j, (a, b, d) in enumerate(edges):
                                if d.get('label') == key:
                                    edge_idx = j
                                    break
                        if edge_idx is None:
                            print("Hrana nenalezena podle indexu ani labelu")
                            continue
                        if u in nodes2 and 0 <= edge_idx < len(edges):
                            i = nodes2.index(u)
                            clear_screen(); print(f"Incidence[{u}, edge_{edge_idx}] = {inc[i][edge_idx]} (edge={edges[edge_idx][0]}-{edges[edge_idx][1]}, label={edges[edge_idx][2].get('label')})")
                        else:
                            print("Neplatný uzel nebo index hrany")
                    elif sub == '2':
                        cntq = input("Zadejte hodnotu pro spočítání výskytu v matici incidence (číslo nebo 'inf'): ").strip()
                        if cntq:
                            try:
                                target = float(cntq) if cntq != 'inf' else 'inf'
                            except Exception:
                                target = cntq
                            c = count_in_matrix(inc, nodes=nodes2, edges=edges, target=target)
                            clear_screen(); print(f"Počet hodnot {cntq} v matici incidence: {c}")
                    elif sub == '3':
                        rc = row_counts(inc)
                        clear_screen(); print(f"Počet nenulových v každém řádku: {dict(zip(nodes2, rc))}")
                    elif sub == '4':
                        cc = col_counts(inc)
                        col_names = [f"edge_{i}" for i in range(len(edges))]
                        clear_screen(); print(f"Počet nenulových v každém sloupci: {dict(zip(col_names, cc))}")
                    elif sub == '5':
                        path = input("Cesta k výstupnímu souboru (např. incidence.txt): ").strip() or 'incidence_matrix.txt'
                        ok, msg = export_matrix_to_file(inc, row_names=nodes2, col_names=[f"edge_{i}" for i in range(len(edges))], path=path)
                        if ok:
                            print(f"Vystup ulozen do: {msg}")
                        else:
                            print(f"Chyba při exportu: {msg}")
                    else:
                        continue
                # --- distance matrix ---
                elif rc == '3':
                    try:
                        import math
                        nodes_d = sorted(list(G.nodes()))
                        n = len(nodes_d)
                        dmat = [[math.inf]*n for _ in range(n)]
                        for i in range(n): dmat[i][i] = 0
                        for u, v, meta in G.edges(data=True):
                            i = nodes_d.index(u); j = nodes_d.index(v)
                            w = meta.get('weight') if meta and meta.get('weight') is not None else 1
                            dmat[i][j] = w
                            if not G.is_directed():
                                dmat[j][i] = w
                        for k2 in range(n):
                            for i in range(n):
                                for j in range(n):
                                    if dmat[i][k2] + dmat[k2][j] < dmat[i][j]:
                                        dmat[i][j] = dmat[i][k2] + dmat[k2][j]
                    except Exception as e:
                        print("Nelze spočítat matici délek:", e); continue
                    print("  1. Hodnota na pozici")
                    print("  2. Počet hodnot")
                    print("  3. Počet hodnot v řádku")
                    print("  4. Počet hodnot ve sloupci")
                    print("  5. Exportovat matici do textového souboru")
                    print("  6. Zpět")
                    sub = get_input("Volba: ")
                    if sub == '1':
                        pos = input("Zadejte uzel zdroj a cílový uzel (např. A B): ").strip()
                        if pos:
                            a, b = pos.split()[:2]
                            if a in nodes_d and b in nodes_d:
                                i = nodes_d.index(a); j = nodes_d.index(b)
                                val = dmat[i][j]
                                clear_screen(); print(f"Vzdálenost {a} -> {b} = {val if val < float('inf') else 'inf'}")
                            else:
                                print("Neznámé uzly")
                    elif sub == '2':
                        cntq = input("Zadejte hodnotu pro spočítání výskytu v matici délek (číslo nebo 'inf'): ").strip()
                        if cntq:
                            try:
                                target = 'inf' if cntq == 'inf' else float(cntq)
                            except Exception:
                                target = cntq
                            c = count_in_matrix(dmat, nodes=nodes_d, target=target)
                            clear_screen(); print(f"Počet hodnot {cntq} v matici délek: {c}")
                    elif sub == '3':
                        rc = row_counts(dmat)
                        clear_screen(); print(f"Počet nenulových v každém řádku: {dict(zip(nodes_d, rc))}")
                    elif sub == '4':
                        cc = col_counts(dmat)
                        clear_screen(); print(f"Počet nenulových v každém sloupci: {dict(zip(nodes_d, cc))}")
                    elif sub == '5':
                        path = input("Cesta k výstupnímu souboru (např. distance.txt): ").strip() or 'distance_matrix.txt'
                        ok, msg = export_matrix_to_file(dmat, row_names=nodes_d, col_names=nodes_d, path=path)
                        if ok:
                            print(f"Vystup ulozen do: {msg}")
                        else:
                            print(f"Chyba při exportu: {msg}")
                    else:
                        continue
                # --- predecessors matrix ---
                elif rc == '4':
                    preds = {}
                    nodes_p = sorted(list(G.nodes()))
                    try:
                        for s in nodes_p:
                            paths = nx.single_source_shortest_path(G, s) if not is_weighted(G) else nx.single_source_dijkstra_path(G, s, weight='weight')
                            for t, p in paths.items():
                                if len(p) >= 2:
                                    preds.setdefault((s, t), p[-2])
                                else:
                                    preds.setdefault((s, t), None)
                    except Exception:
                        pass
                    print("  1. Hodnota na pozici (A B)")
                    print("  2. Počet hodnot (není podporováno)")
                    print("  3. Exportovat tabulku predchudcu do souboru")
                    print("  4. Zpět")
                    sub = get_input("Volba: ")
                    if sub == '1':
                        pos = input("Zadejte A B: ").strip()
                        if pos:
                            a, b = pos.split()[:2]
                            clear_screen(); print(f"Předchůdce na cestě {a}->{b}: {preds.get((a,b))}")
                    elif sub == '3':
                        path = input("Cesta k souboru pro export tabulky předchůdců: ").strip() or 'predecessors_table.txt'
                        try:
                            with open(path, 'w', encoding='utf-8') as fh:
                                for k, v in preds.items():
                                    fh.write(f"{k[0]}\t{k[1]}\t{v}\n")
                            print(f"Vystup ulozen do: {path}")
                        except Exception as e:
                            print("Chyba při exportu:", e)
                    else:
                        continue
                # --- edge table ---
                elif rc == '5':
                    rows = []
                    for j, (u, v, d) in enumerate(G.edges(data=True)):
                        rows.append((j, f"{u}-{v}", u, v, d.get('weight'), d.get('label')))
                    print("  1. Hodnota na pozici (index nebo label)")
                    print("  2. Počet hodnot (není podporováno)")
                    print("  3. Exportovat tabulku hran do souboru")
                    print("  4. Zpět")
                    sub = get_input("Volba: ")
                    if sub == '1':
                        q = input("Zadejte index hrany nebo label (např. 0 nebo e1 nebo :e1): ").strip()
                        if not q:
                            continue
                        key = q[1:] if q.startswith(':') else q
                        sel_idx = None
                        try:
                            sel_idx = int(key)
                        except Exception:
                            for j, name, u, v, w, lab in rows:
                                if lab == key or name == key:
                                    sel_idx = j; break
                        if sel_idx is None:
                            print("Hrana nenalezena")
                            continue
                        for j, name, u, v, w, lab in rows:
                            if j == sel_idx:
                                clear_screen(); print(f"Hrana {j}: {u}-{v} (váha={w}, label={lab})")
                                break
                    elif sub == '3':
                        path = input("Cesta k souboru pro export tabulky hran: ").strip() or 'edges_table.txt'
                        try:
                            with open(path, 'w', encoding='utf-8') as fh:
                                fh.write('idx\tu\tv\tweight\tlabel\n')
                                for j, name, u, v, w, lab in rows:
                                    fh.write(f"{j}\t{u}\t{v}\t{w}\t{lab}\n")
                            print(f"Vystup ulozen do: {path}")
                        except Exception as e:
                            print("Chyba při exportu:", e)
                    else:
                        continue

            # 3. Node analysis
            elif choice == '3':
                node = input("Zadejte jméno uzlu: ").strip()
                if not node:
                    print("Nezadán uzel")
                    continue
                print("  1. Následníci")
                print("  0. Vypsat vše")
                print("  2. Předchůdci")
                print("  3. Sousedé")
                print("  4. Výstupní okolí")
                print("  5. Vstupní okolí")
                print("  6. Okolí")
                print("  7. Výstupní stupeň")
                print("  8. Vstupní stupeň")
                print("  9. Stupeň")
                print("  10. Vstupní uzly")
                print("  11. Výstupní uzly")
                print("  12. Incidenční uzly")
                nc = get_input("Volba: ")
                if nc == '0':
                    # vypsat vše pro uzel
                    clear_screen()
                    print(f"Následníci: {node_successors(G, node)}")
                    print(f"Předchůdci: {node_predecessors(G, node)}")
                    print(f"Sousedé: {node_neighbors(G, node)}")
                    print(f"Výstupní okolí: {out_neighborhood(G, node)}")
                    print(f"Vstupní okolí: {in_neighborhood(G, node)}")
                    print(f"Okolí: {neighborhood(G, node)}")
                    print(f"Výstupní stupeň: {out_degree_node(G, node)}")
                    print(f"Vstupní stupeň: {in_degree_node(G, node)}")
                    print(f"Stupeň: {degree_node(G, node)}")
                    print(f"Vstupní uzly: {[n for n in G.nodes() if G.in_degree(n) > 0] if G.is_directed() else []}")
                    print(f"Výstupní uzly: {[n for n in G.nodes() if G.out_degree(n) > 0] if G.is_directed() else []}")
                    incident = set()
                    for u, v, d in G.edges(data=True):
                        if u == node or v == node:
                            incident.add(u); incident.add(v)
                    print(f"Incidenční uzly: {sorted(list(incident))}")
                elif nc == '1':
                    clear_screen()
                    print(node_successors(G, node))
                elif nc == '2':
                    print(node_predecessors(G, node))
                elif nc == '3':
                    print(node_neighbors(G, node))
                elif nc == '4':
                    print(out_neighborhood(G, node))
                elif nc == '5':
                    print(in_neighborhood(G, node))
                elif nc == '6':
                    print(neighborhood(G, node))
                elif nc == '7':
                    print(out_degree_node(G, node))
                elif nc == '8':
                    print(in_degree_node(G, node))
                elif nc == '9':
                    print(degree_node(G, node))
                elif nc == '10':
                    print([n for n in G.nodes() if G.in_degree(n) > 0] if G.is_directed() else [])
                elif nc == '11':
                    print([n for n in G.nodes() if G.out_degree(n) > 0] if G.is_directed() else [])
                elif nc == '12':
                    # incidenční uzly: nodes incident to same edges
                    incident = set()
                    for u, v, d in G.edges(data=True):
                        if u == node or v == node:
                            incident.add(u); incident.add(v)
                    clear_screen()
                    print(sorted(list(incident)))
                else:
                    print("Neplatná volba")

            # 4. Edge analysis
            elif choice == '4':
                edges_list = list(G.edges(data=True))
                if not edges_list:
                    print("Graf nemá žádné hrany")
                    continue
                print("Seznam hran:")
                for i, (u, v, d) in enumerate(edges_list):
                    print(f"  {i}. {u} - {v} (váha={d.get('weight')}, label={d.get('label')})")
                print("Zadejte index hrany nebo B pro návrat.")
                ei = get_input("Vyberte index hrany: ")
                if ei.lower() == 'b' or ei == '':
                    continue
                try:
                    ei = int(ei)
                except Exception:
                    print("Neplatný index")
                    continue
                if ei < 0 or ei >= len(edges_list):
                    print("Index mimo rozsah")
                    continue
                u, v, d = edges_list[ei]
                print(f"Hrana {ei}: {u} -> {v}")
                print(f"  váha: {d.get('weight')}")
                print(f"  label: {d.get('label')}")
                print(f"  smyčka: {u==v}")
                print(f"  orientovaná: {G.is_directed()}")
                # incidentní uzly
                clear_screen()
                print(f"  incidentní uzly: {u}, {v}")
                # check if bridge
                try:
                    if not G.is_directed():
                        print(f"  most (bridge): {nx.is_bridge(G, (u,v))}")
                except Exception:
                    pass
                # add 'vypsat vse' option for edge: show all properties
                all_q = input("Chcete vypsat vše o této hraně? [y/N]: ").strip().lower()
                if all_q == 'y':
                    print(f"Detailně: hrana mezi {u} a {v}")
                    print(f"  váha: {d.get('weight')}")
                    print(f"  label: {d.get('label')}")
                    print(f"  smyčka: {u==v}")
                    print(f"  orientovaná: {G.is_directed()}")
                    print(f"  incidentní uzly: {u}, {v}")

            # 5. Operace nad grafem
            elif choice == '5':
                print("Operace:")
                print("  1. Nejkratší cesta mezi dvěma uzly")
                print("  2. Minimální kostra (neorientovaný pohled)")
                print("  3. Komponenty")
                print("  4. Topologické řazení (pokud orientovaný)")
                print("  5. Detekce cyklů")
                print("  6. Centrality (top K)")
                oc = get_input("Volba: ")
                if oc == '1':
                    s = input("Počáteční uzel: ").strip(); t = input("Cílový uzel: ").strip()
                    class A: pass
                    a = A(); a.file = current_file; a.src = s; a.dst = t
                    cmd_shortest(a)
                elif oc == '2':
                    class A: pass
                    a = A(); a.file = current_file
                    cmd_mst(a)
                elif oc == '3':
                    class A: pass
                    a = A(); a.file = current_file
                    cmd_components(a)
                elif oc == '4':
                    class A: pass
                    a = A(); a.file = current_file
                    cmd_topo(a)
                elif oc == '5':
                    class A: pass
                    a = A(); a.file = current_file
                    cmd_cycles(a)
                elif oc == '6':
                    k = get_input("Kolik prvků vypsat (výchozí 5): ")
                    try:
                        k = int(k) if k else 5
                    except Exception:
                        k = 5
                    class A: pass
                    a = A(); a.file = current_file; a.top = k
                    cmd_centrality(a)
                else:
                    print("Neplatná volba")
            else:
                print("Neplatná volba v hlavní nabídce")
        return
    if args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "shortest":
        cmd_shortest(args)
    elif args.cmd == "mst":
        cmd_mst(args)
    elif args.cmd == "centrality":
        cmd_centrality(args)
    elif args.cmd == "components":
        cmd_components(args)
    elif args.cmd == "cycles":
        cmd_cycles(args)
    elif args.cmd == "topo":
        cmd_topo(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
