import copy
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.drawing import nx_pydot
import re
import numpy as np
from scipy.optimize import curve_fit
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import linregress


class WWWGraph:
    def __init__(self, pages_limit):
        self.G = nx.DiGraph()
        self.pages_limit = pages_limit
        self.pages_num = {}
        self.label = ""

    def addDirectedEdge(self, page, sub_page):
        self.G.add_edge(self.format_link(page), self.format_link(sub_page))

    def format_link(self, link):
        return (
            re.sub(r"https?://", "", link)
            .replace("/", "-")
            .replace(".", "-")
            .replace("%", "-")
        )

    def read(self):
        self.G = nx.read_gml("graph.gml", label="id")

    def save(self):
        nx.write_gml(self.G, "graph.gml")

    def nodes(self):
        return self.G.number_of_nodes()

    def edges(self):
        return self.G.number_of_edges()

    def printGraph(self, graph):

        valid_names = {
            node: re.sub(r"https?://", "", node)
            .replace("/", "-")
            .replace(".", "-")
            .replace("%", "-")
            for node in graph.nodes
        }
        graph = nx.relabel_nodes(graph, valid_names)
        pos = nx.spring_layout(graph, k=2.5)
        # pos = nx.spring_layout(graph, k=0.5)
        # pos = nx.spring_layout(graph, k=2)
        # pos = nx.random_layout(graph, seed=random.randint(0, 0xFFFFFFFF))

        for node, _ in pos.items():
            graph.nodes[node][
                "node_options"
            ] = "circle, draw=black, fill=blue,minimum size=2pt"
            graph.nodes[node]["node_label"] = ""
        # to_latex_raw only tkiz
        latex = nx.to_latex_raw(
            graph,
            pos=pos,
            node_label="node_label",
            node_options="node_options",
        )

        latex = latex.replace("\\begin{scope}[->]", "\\begin{scope}[->, opacity=0.1]")
        latex = latex.replace("\\begin{tikzpicture}", "\\begin{tikzpicture}[scale=7]")
        with open("graph1.tex", "w", encoding="utf-8") as f:
            f.write(latex)

    def save_file(self, filename, text):
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    def basic(self, graph):
        print(
            f"[{self.label}] V={graph.number_of_nodes()}, E={graph.number_of_edges()}"
        )

    def InOutDeg(self, graph):
        # rozkłady stopni (in, out), wyznaczenie współczynników funkcji potęgowej metodami analitycznymi (np. regresja),
        in_deg = Counter(Counter(dict(graph.in_degree())).values())
        out_deg = Counter(Counter(dict(graph.out_degree())).values())
        x_in, y_in = zip(*in_deg.items())
        x_out, y_out = zip(*out_deg.items())

        inSort = sorted([i for i in in_deg.keys()])
        self.save_file(
            f"in_deg_{self.label}", "".join([f"({i}, {in_deg[i]})\n" for i in inSort])
        )
        outSort = sorted([i for i in out_deg.keys()])
        self.save_file(
            f"out_deg_{self.label}",
            "".join([f"({i}, {out_deg[i]})\n" for i in outSort]),
        )

        self.print_linregress(x_in, y_in, f"[{self.label}] Wejściowe")
        self.print_linregress(x_out, y_out, f"[{self.label}] Wyjściowe")

    def print_linregress(self, x, y, label):
        non_zero_x = np.array(x)[np.array(x) > 0]
        non_zero_y = np.array(y)[np.array(x) > 0]

        log_x = np.log(non_zero_x)
        log_y = np.log(non_zero_y)

        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        b = slope
        a = np.exp(intercept)
        print(
            f"[{self.label}]  Wzór rozkładu potęgowego ({label}): P(k) = {a:.4f} * x^{b:.4f} R^2 = {r_value**2:.2f}\n"
        )

    def SCCWCC(self, graph):
        # analiza składowych spójności: słabe (WCC), silne (SCC), komponenty IN, OUT, graf SCC (podział na SCC),
        wcc = len(
            list(nx.connected_components(graph.to_undirected()))
        )  # len(list(nx.weakly_connected_components(graph)))
        scc = len(list(nx.strongly_connected_components(graph)))

        print(f"[{self.label}]  Liczba słabych komponentów spójności (WCC): {wcc}")
        print(f"[{self.label}]  Liczba silnych komponentów spójności (SCC): {scc}")

    def INOUTComp(self, G):
        sccs = list(nx.strongly_connected_components(G))
        largest_scc = max(sccs, key=len)

        in_component = set()
        for node in g.nodes:
            for scc_node in largest_scc:
                if nx.has_path(g, node, scc_node):
                    in_component.add(node)
                    break

        out_component = set()
        for node in g.nodes:
            for scc_node in largest_scc:
                if nx.has_path(g, scc_node, node):
                    out_component.add(node)
                    break

        print(f"Liczba węzłów w komponencie IN: {len(in_component)}")
        print(f"Liczba węzłów w komponencie OUT: {len(out_component)}")

    def distances(self, graph):
        shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

        # średnie odległości dla każdej pary
        distances = []
        for source, targets in shortest_paths.items():
            for target, distance in targets.items():
                if source != target:  # Takie same pary
                    distances.append(distance)

        avg_distance = np.mean(distances)
        counter = Counter(distances)
        print(f"[{self.label}]  distances {counter}")
        print(f"[{self.label}]  Średnia odległość w grafie: {avg_distance}")

        self.print_linregress(
            list(counter.keys()), list(counter.values()), f"distances_{self.label}"
        )

        x = np.array(list(counter.keys()))
        y = np.array(list(counter.values()))

        # Regresja kwadratowa: y = a*x^2 + b*x + c
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        print(
            f"[{self.label}] distances  Regresja kwadratowa: {a:.4f} * x2 + {b:.4f} * x + {c:.4f}"
        )

    def diameter(self, graph):
        # Średnica grafu (maksymalna odległość)
        diameter = nx.diameter(graph)
        print(f"[{self.label}]  Średnica grafu: {diameter}")

    def radius(self, graph):
        # Promień grafu (minimalna odległość do najdalszego wierzchołka)
        radius = nx.radius(graph)
        print(f"[{self.label}] Promień grafu: {radius}")

    def cluster(self, graph):
        # współczynniki klasteryzacji: lokalne oraz globalne (analiza histogramów i regresja dla rozkładów)
        # Lokalny współczynnik klasteryzacji
        local_clustering = nx.clustering(graph)

        # Globalny współczynnik klasteryzacji
        # nx.average_clustering(graph) wychodzi to samo co local_clustering
        global_clustering = nx.transitivity(graph.to_undirected())

        print(
            f"[{self.label}] Lokalny współczynnik klasteryzacji: {np.mean(list(local_clustering.values()))}"
        )
        print(
            f"[{self.label}] Globalny współczynnik klasteryzacji: {global_clustering}"
        )
        counter = Counter(round(v, 2) for v in local_clustering.values())
        self.print_linregress(
            list(counter.keys()), list(counter.values()), f"[{self.label}] localclust2"
        )
        self.save_file(
            f"localcluster2_{self.label}",
            "".join(
                f"({k}, {v})\n"
                for k, v in Counter(
                    round(v, 2) for v in local_clustering.values()
                ).items()
            ),
        )

    def vertex_connectivityNP(self, graph):
        try:
            print("[INFO] Start node_connectivity obliczeń...")
            k = nx.node_connectivity(graph)
            print(f"[OK] Vertex connectivity (node_connectivity) = {k}")
        except Exception as e:
            print(e)

    def vertex_connectivity(self, graph):
        print("[INFO] Start node_connectivity obliczeń...")
        # k = nx.node_connectivity(graph)
        undir = self.G.to_undirected()
        pairs = list(nx.all_node_cuts(undir, 2))
        print(f"slabe {pairs}, len {len(pairs)}")
        # print(f"[OK] Vertex connectivity (node_connectivity) = {k}")

    def excentricity(self, graph):
        eccentricities = nx.eccentricity(graph)
        ecc_counts = Counter(eccentricities.values())
        print(ecc_counts)

    def custom_pagerank(self, graph, alpha):
        tol = 1e-6
        max_iter = 100
        N = graph.number_of_nodes()
        nodes = list(graph.nodes())
        ranks = {node: 1 / N for node in nodes}
        for _ in range(max_iter):
            new_ranks = {}
            for node in nodes:
                rank_sum = sum(
                    ranks[neigh] / graph.out_degree(neigh)
                    for neigh in graph.predecessors(node)
                    if graph.out_degree(neigh) > 0
                )
                new_ranks[node] = (1 - alpha) / N + alpha * rank_sum
            delta = sum(abs(new_ranks[n] - ranks[n]) for n in nodes)
            ranks = new_ranks
            if delta < tol:
                break
        counter = Counter(round(v, 7) for v in ranks.values())
        # print(counter)
        self.print_linregress(
            list(counter.keys()), list(counter.values()), f"[{self.label}] pg a={alpha}"
        )
        self.save_file(
            f"pgas a={alpha}_{self.label}",
            "".join(f"({k:.7f}, {v})\n" for k, v in counter.items()),
        )
        print(f"{min(list(counter.keys())):.7f}, {max(list(counter.keys())):.7f}")

    def run_custom_pagerank(self, graph):
        print("\n[INFO] Analiza zbieżności przy różnych wartościach tłumienia:")
        for d in [1, 0.85, 0.55]:
            print(f"\nalfa = {d}")
            self.custom_pagerank(graph, alpha=d)

    def simulate_failure_and_attack_scenarios(self, Graph, idx, seed=None):

        def remove_nodes(g, nodes_to_remove):
            g_copy = copy.deepcopy(g)
            g_copy.remove_nodes_from(nodes_to_remove)
            return g_copy

        random.seed(seed)
        n = Graph.number_of_nodes()
        nodes = list(Graph.nodes)

        degree_sorted = sorted(Graph.degree(), key=lambda x: x[1], reverse=True)
        attack_10 = [node for node, _ in degree_sorted[: int(0.10 * n)]]
        attack_20 = [node for node, _ in degree_sorted[: int(0.20 * n)]]
        attack_50 = [node for node, _ in degree_sorted[: int(0.50 * n)]]
        failure_10 = random.sample(nodes, int(0.10 * n))
        failure_20 = random.sample(nodes, int(0.20 * n))
        failure_30 = random.sample(nodes, int(0.30 * n))
        failure_50 = random.sample(nodes, int(0.50 * n))
        scenarios = None

        if 1 == idx:
            scenarios = remove_nodes(Graph, [nodes[0]])
        if 2 == idx:
            scenarios = remove_nodes(Graph, failure_10)
        if 3 == idx:
            scenarios = remove_nodes(Graph, failure_20)
        if 4 == idx:
            scenarios = remove_nodes(Graph, failure_30)
        if 5 == idx:
            scenarios = remove_nodes(Graph, failure_50)
        if 6 == idx:
            scenarios = remove_nodes(Graph, attack_10)
        if 7 == idx:
            scenarios = remove_nodes(Graph, attack_20)
        if 8 == idx:
            scenarios = remove_nodes(Graph, attack_50)

        return scenarios

    def analyze_graph(self):
        workers = 4
        functions = [
            # self.basic,
            # self.SCCWCC,
            # self.InOutDeg,
            # self.distances,
            # self.cluster,
            # self.diameter,
            # self.radius,
            # self.run_custom_pagerank,
            self.vertex_connectivity,
        ]
        for idx in range(1, 9):
            try:
                print(f"===============tryb {idx}=====================")
                self.label = f"tryb_{idx}"
                graph = self.simulate_failure_and_attack_scenarios(self.G, idx, 23)
                self.excentricity(graph)
            except Exception as e:
                print(e)
                # futures = [executor.submit(func, graph) for func in functions]
                # for future in futures:
                #    future.result()
            # with ThreadPoolExecutor(max_workers=workers) as executor:
            #     futures = [executor.submit(func, self.G) for func in functions]
            #     for future in futures:
            #         future.result()


if __name__ == "__main__":

    g = WWWGraph(1)
    g.read()
    g.analyze_graph()
    # g.run_custom_pagerank(g.G)
