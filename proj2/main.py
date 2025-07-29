from lsi_clustering import lsi_clustering
from graph_analysis import build_similarity_graph, analyze_graph
from utils import load_documents_from_html
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Załaduj i przetwórz dokumenty HTML
folder_path = "./download"
texts, filenames = load_documents_from_html(folder_path)

# 2. LSI + klasteryzacja
labels, X_lsi, silhouette, model = lsi_clustering(texts, n_topics=100, n_clusters=5)
print(f"Silhouette Score: {silhouette:.3f}")

# 3. Analiza grafowa
G = build_similarity_graph(X_lsi, filenames, threshold=0.7)
pagerank, communities, modularity = analyze_graph(G)
print(f"Modularity: {modularity:.3f}")


# 4. Wizualizacja klastrów
def plot_clusters(X, labels, filenames):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(set(labels)):
        plt.scatter(
            reduced[labels == label, 0],
            reduced[labels == label, 1],
            label=f"Cluster {label}",
            alpha=0.7,
        )
    for i, name in enumerate(filenames):
        plt.text(reduced[i, 0], reduced[i, 1], name, fontsize=8, alpha=0.5)
    plt.legend()
    plt.title("Dokumenty w przestrzeni LSI")
    plt.show()


plot_clusters(X_lsi, labels, filenames)
