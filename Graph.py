import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing import nx_pydot
import re
import numpy as np
from scipy.optimize import curve_fit
import random
from collections import Counter


class WWWGraph:
    def __init__(self, pages_limit):
        self.G = nx.DiGraph()
        self.pages_limit = pages_limit

    def addDirectedEdge(self, page, sub_page):
        self.G.add_edge(page, sub_page)

    def nodes(self):
        return self.G.number_of_nodes()

    def edges(self):
        return self.G.number_of_edges()

    def read(self):
        self.G = nx.read_gml("graph.gml")
        # self.G = self.G.subgraph(list(self.G.nodes)[:3000]).copy()

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
        pos = nx.spring_layout(self.G, k=2)
        # pos = nx.spring_layout(self.G, k=0.5)
        # pos = nx.spring_layout(self.G, k=2)
        # pos = nx.random_layout(self.G, seed=random.randint(0, 0xFFFFFFFF))

        for node, _ in pos.items():
            self.G.nodes[node][
                "node_options"
            ] = "circle, draw=black, fill=gray,minimum size=2pt"
            self.G.nodes[node]["node_label"] = ""
        # to_latex_raw only tkiz
        latex = nx.to_latex(
            self.G,
            pos=pos,
            node_label="node_label",
            node_options="node_options",
        )

        latex = latex.replace("\\begin{scope}[->]", "\\begin{scope}[->, opacity=0.12]")
        latex = latex.replace("\\begin{tikzpicture}", "\\begin{tikzpicture}[scale=15]")
        with open("graph.tex", "w", encoding="utf-8") as f:
            f.write(latex)

    def power_law(self, x, a, b):
        return a * np.power(x, b)

    def analyze_graph(self):
        stats = {}

        # 1. Liczba wierzchołków i łuków
        stats["nodes"] = self.G.number_of_nodes()
        stats["edges"] = self.G.number_of_edges()
        # stats["inDEG"] = self.G.in_degree()
        # stats["outDEG"] = self.G.out_degree()
        # 2. Analiza składowych spójności
        stats["WCC"] = len(list(nx.weakly_connected_components(self.G)))
        stats["SCC"] = len(list(nx.strongly_connected_components(self.G)))

        # 3. Rozkłady stopni (in-degree, out-degree)
        stats["inDEG"] = Counter(dict(self.G.in_degree()))
        stats["ourDEG"] = Counter(dict(self.G.out_degree()))

        # 4. Najkrótsze ścieżki i ich analiza
        if nx.is_strongly_connected(self.G):
            lengths = dict(nx.all_pairs_shortest_path_length(self.G))
            all_lengths = [
                length for paths in lengths.values() for length in paths.values()
            ]
            stats["avg_shortest_path"] = np.mean(all_lengths)
            stats["graph_diameter"] = np.max(all_lengths)
            stats["radius"] = np.min(all_lengths)
            stats["shortest_path_hist"] = np.histogram(all_lengths, bins=20)[0].tolist()

            # Zapis histogramu długości ścieżek
            plt.hist(all_lengths, bins=20, alpha=0.7, color="g")
            plt.xlabel("Długość ścieżki")
            plt.ylabel("Liczba par wierzchołków")
            plt.title("Histogram długości najkrótszych ścieżek")
            plt.savefig("shortest_path_histogram.png")
            plt.clf()

        # 5. Współczynniki klasteryzacji
        stats["global_clustering"] = nx.average_clustering(self.G.to_undirected())

        # 6. Odporność na awarie i ataki
        def analyze_resilience(G, remove_func):
            G_copy = G.copy()
            removed_nodes = remove_func(G_copy)
            new_WCC = len(list(nx.weakly_connected_components(G_copy)))
            new_SCC = len(list(nx.strongly_connected_components(G_copy)))
            return {"removed_nodes": len(removed_nodes), "WCC": new_WCC, "SCC": new_SCC}

        def random_failures(G):
            nodes_to_remove = np.random.choice(
                G.nodes(), size=int(0.1 * G.number_of_nodes()), replace=False
            )
            G.remove_nodes_from(nodes_to_remove)
            return nodes_to_remove

        def targeted_attacks(G):
            high_degree_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[
                : int(0.1 * G.number_of_nodes())
            ]
            nodes_to_remove = [n for n, _ in high_degree_nodes]
            G.remove_nodes_from(nodes_to_remove)
            return nodes_to_remove

        stats["resilience_random"] = analyze_resilience(self.G, random_failures)
        stats["resilience_targeted"] = analyze_resilience(self.G, targeted_attacks)

        # 7. Spójność wierzchołkowa
        stats["articulation_points"] = len(
            list(nx.articulation_points(self.G.to_undirected()))
        )

        return stats


if __name__ == "__main__":

    g = WWWGraph(1)
    g.read()
    # g.printGraph()
    stats = g.analyze_graph()
    for key, value in stats.items():
        print(f"{key}: {value}")
