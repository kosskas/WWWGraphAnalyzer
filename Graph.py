import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing import nx_pydot
import re
import random


class WWWGraph:
    def __init__(self, pages_limit):
        self.G = nx.DiGraph()
        self.pages_limit = pages_limit

    def addDirectedEdge(self, page, sub_page):
        self.G.add_edge(page, sub_page)

    def basic_stats(self):
        """Zwraca podstawowe statystyki grafu"""
        num_nodes = len(self.G.nodes())
        num_edges = len(self.G.edges())
        return num_nodes, num_edges

    def printGraph(self):

        valid_names = {
            node: re.sub(r"https?://", "", node)
            .replace("/", "-")
            .replace(".", "-")
            .replace("%", "-")
            for node in self.G.nodes
        }
        self.G = nx.relabel_nodes(self.G, valid_names)
        pos = nx.spring_layout(self.G)
        # pos = nx.spring_layout(self.G, k=0.5)
        # pos = nx.spring_layout(self.G, k=2)
        # pos = nx.random_layout(self.G, seed=random.randint(0, 0xFFFFFFFF))

        for node, _ in pos.items():
            self.G.nodes[node][
                "node_options"
            ] = "circle, draw=black, fill=blue,minimum size=2pt"
            self.G.nodes[node]["node_label"] = ""
        # to_latex_raw only tkiz
        latex = nx.to_latex(
            self.G,
            pos=pos,
            node_label="node_label",
            node_options="node_options",
        )

        latex = latex.replace("\\begin{scope}[->]", "\\begin{scope}[->, opacity=0.2]")
        latex = latex.replace("\\begin{tikzpicture}", "\\begin{tikzpicture}[scale=15]")
        with open("graph.tex", "w", encoding="utf-8") as f:
            f.write(latex)
