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
        self.G = nx.read_gml("graph.gml")

    def save(self):
        nx.write_gml(self.G, "graph.gml")

    def printGraph(self):

        valid_names = {
            node: re.sub(r"https?://", "", node)
            .replace("/", "-")
            .replace(".", "-")
            .replace("%", "-")
            for node in self.G.nodes
        }
        self.G = nx.relabel_nodes(self.G, valid_names)
        pos = nx.spring_layout(self.G, k=2.5)
        # pos = nx.spring_layout(self.G, k=0.5)
        # pos = nx.spring_layout(self.G, k=2)
        # pos = nx.random_layout(self.G, seed=random.randint(0, 0xFFFFFFFF))

        for node, _ in pos.items():
            self.G.nodes[node][
                "node_options"
            ] = "circle, draw=black, fill=blue,minimum size=2pt"
            self.G.nodes[node]["node_label"] = ""
        # to_latex_raw only tkiz
        latex = nx.to_latex_raw(
            self.G,
            pos=pos,
            node_label="node_label",
            node_options="node_options",
        )

        latex = latex.replace("\\begin{scope}[->]", "\\begin{scope}[->, opacity=0.1]")
        latex = latex.replace("\\begin{tikzpicture}", "\\begin{tikzpicture}[scale=7]")
        with open("graph.tex", "w", encoding="utf-8") as f:
            f.write(latex)

    def extract_subgraph(graph, node):
        # N and N
        neighbors = set(graph.neighbors(node))
        second_neighbors = set()

        for neighbor in neighbors:
            second_neighbors.update(graph.neighbors(neighbor))
        relevant_nodes = {node} | neighbors | second_neighbors

        subgraph = graph.subgraph(relevant_nodes).copy()

        return subgraph

    def save_file(self, filename, text):
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    def basic(self):
        print(f"V={self.G.number_of_nodes()}, E={self.G.number_of_edges()}")

    def InOutDeg(self):
        # rozkłady stopni (in, out), wyznaczenie współczynników funkcji potęgowej metodami analitycznymi (np. regresja),
        in_deg = Counter(Counter(dict(self.G.in_degree())).values())
        out_deg = Counter(Counter(dict(self.G.out_degree())).values())
        x_in, y_in = zip(*in_deg.items())
        x_out, y_out = zip(*out_deg.items())

        inSort = sorted([i for i in in_deg.keys()])
        self.save_file("in_deg", "".join([f"({i}, {in_deg[i]})\n" for i in inSort]))
        outSort = sorted([i for i in out_deg.keys()])
        self.save_file("out_deg", "".join([f"({i}, {out_deg[i]})\n" for i in outSort]))

        self.print_linregress(x_in, y_in, "Wejściowe")
        self.print_linregress(x_out, y_out, "Wyjściowe")

    def print_linregress(self, x, y, label):
        non_zero_x = np.array(x)[np.array(x) > 0]
        non_zero_y = np.array(y)[np.array(x) > 0]

        log_x = np.log(non_zero_x)
        log_y = np.log(non_zero_y)

        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        b = slope
        a = np.exp(intercept)
        print(f"Wzór rozkładu potęgowego ({label}):")
        print(f"P(k) = {a:.4f} * x^{b:.4f}")
        print(f"R^2 = {r_value**2:.2f}\n")

    def SCCWCC(self):
        # analiza składowych spójności: słabe (WCC), silne (SCC), komponenty IN, OUT, graf SCC (podział na SCC),
        wcc = len(
            list(nx.connected_components(self.G.to_undirected()))
        )  # len(list(nx.weakly_connected_components(self.G)))
        scc = len(list(nx.strongly_connected_components(self.G)))

        print(f"Liczba słabych komponentów spójności (WCC): {wcc}")
        print(f"Liczba silnych komponentów spójności (SCC): {scc}")

    def distances(self):
        shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))

        # średnie odległości dla każdej pary
        distances = []
        for source, targets in shortest_paths.items():
            for target, distance in targets.items():
                if source != target:  # Takie same pary
                    distances.append(distance)

        avg_distance = np.mean(distances)
        counter = Counter(distances)
        print(f"distances {counter}")
        print(f"Średnia odległość w grafie: {avg_distance}")

        self.print_linregress(list(counter.keys()), list(counter.values()), "distances")

    def diameter(self):
        # Średnica grafu (maksymalna odległość)
        diameter = nx.diameter(self.G)
        print(f"Średnica grafu: {diameter}")

    def radius(self):
        # Promień grafu (minimalna odległość do najdalszego wierzchołka)
        radius = nx.radius(self.G)
        print(f"Promień grafu: {radius}")

    def cluster(self):
        # współczynniki klasteryzacji: lokalne oraz globalne (analiza histogramów i regresja dla rozkładów)
        # Lokalny współczynnik klasteryzacji
        local_clustering = nx.clustering(self.G)

        # Globalny współczynnik klasteryzacji
        # nx.average_clustering(self.G) wychodzi to samo co local_clustering
        global_clustering = nx.transitivity(self.G.to_undirected())

        print(
            f"Lokalny współczynnik klasteryzacji: {np.mean(list(local_clustering.values()))}"
        )
        print(f"Globalny współczynnik klasteryzacji: {global_clustering}")
        counter = Counter(round(v, 2) for v in local_clustering.values())
        self.print_linregress(
            list(counter.keys()), list(counter.values()), "localclust"
        )
        self.save_file(
            "localcluster",
            "".join(f"({k}, {v})\n" for k, v in counter.items()),
        )

    def pagerank(self):
        pagerank = nx.pagerank(self.G, alpha=0.85)
        # Rozkład PageRank
        pr_values = list(pagerank.values())
        plt.hist(pr_values, bins=20, alpha=0.5, label="PageRank")
        plt.savefig("pagerank.png")

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
            self.pagerank,
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(func) for func in functions]
            for future in futures:
                future.result()


if __name__ == "__main__":

    g = WWWGraph(1)
    g.read()
    g.analyze_graph()
