import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_graph(X_lsi, filenames, threshold=0.7):
    sim_matrix = cosine_similarity(X_lsi)
    G = nx.Graph()
    for i, name_i in enumerate(filenames):
        for j, name_j in enumerate(filenames):
            if i < j and sim_matrix[i][j] > threshold:
                G.add_edge(name_i, name_j, weight=sim_matrix[i][j])
    return G


def analyze_graph(G):
    pagerank = nx.pagerank(G)
    communities = list(nx.community.greedy_modularity_communities(G))
    modularity = nx.community.modularity(G, communities)
    return pagerank, communities, modularity
