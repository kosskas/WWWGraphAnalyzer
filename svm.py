import os
from bs4 import BeautifulSoup
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    silhouette_score,
    adjusted_rand_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# === 1. Konfiguracja ===
DIR = "./download"
USE_LABELS = False

print("🔧 Konfiguracja ustawiona.")
print(f"📁 Katalog z plikami: {DIR}")
print(
    f"🏷️  Tryb: {'Klasyfikacja (z etykietami)' if USE_LABELS else 'Grupowanie (bez etykiet)'}"
)

# === 2. Ładowanie języka polskiego ===
print("⏳ Ładowanie modelu języka polskiego SpaCy...")
nlp = spacy.load("pl_core_news_sm")
print("✅ Model załadowany.")


# === 3. Ekstrakcja tekstu z HTML ===
def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()


def preprocess(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)


print("📥 Rozpoczynam ekstrakcję i preprocessing dokumentów...")
texts = []
filenames = []

total_files = len(os.listdir(DIR))
for i, filename in enumerate(os.listdir(DIR), start=1):
    path = os.path.join(DIR, filename)
    raw_text = extract_text_from_html(path)
    processed = preprocess(raw_text)
    if processed.strip():
        texts.append(processed)
        filenames.append(filename)

    print(f"  ➤ Przetworzono {i}/{total_files} plików...")
    # if i == 100:
    #     print(f"  ➤ Koniec")
    #     break

print(f"✅ Przetworzono {len(texts)} dokumentów.\n")

# === 4. TF-IDF ===
print("🔢 Przekształcanie tekstów do wektorów TF-IDF...")
vectorizer = TfidfVectorizer(max_df=1.0, min_df=10)
X = vectorizer.fit_transform(texts)
print(f"✅ Macierz TF-IDF ma rozmiar: {X.shape}\n")

# === 5. Kategoryzacja (jeśli masz etykiety) ===
if USE_LABELS:
    labels = [...]  # ← Wstaw listę etykiet

    print("🤖 Trening klasyfikatora SVM...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== 📊 Wyniki klasyfikacji (SVM) ===")
    print(classification_report(y_test, y_pred))
else:
    # === 6. Grupowanie ===
    print("🔗 Grupowanie dokumentów metodą KMeans...")
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)

    score = silhouette_score(X, clusters)
    print(f"✅ Grupowanie zakończone. Silhouette Score: {score:.3f}\n")

# === 7. Budowa grafu podobieństwa ===
print("🧠 Buduję graf podobieństw między dokumentami...")
similarity = cosine_similarity(X)
G = nx.Graph()

for i in range(len(texts)):
    G.add_node(i, label=filenames[i])
    for j in range(i + 1, len(texts)):
        if similarity[i, j] > 0.3:
            G.add_edge(i, j, weight=similarity[i, j])

print(
    f"✅ Graf utworzony: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi.\n"
)

# === 8. Analiza grafowa ===
print("📈 Obliczanie PageRank i centralności...")
pagerank = nx.pagerank(G)
centrality = nx.degree_centrality(G)

top_docs = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n=== ⭐ Top 5 dokumentów wg PageRank ===")
for idx, score in top_docs:
    print(f"{filenames[idx]}: {score:.4f}")

# === 9. Ewaluacja jakości klastra ===
if USE_LABELS:
    print("\n📊 Ewaluacja zgodności z etykietami...")
    ari = adjusted_rand_score(labels, clusters)
    vmeasure = v_measure_score(labels, clusters)
    print(f"\nARI: {ari:.3f}, V-Measure: {vmeasure:.3f}")

# === 10. Wizualizacja 2D (t-SNE) ===
print("\n🎨 Generowanie wizualizacji 2D (t-SNE)...")
X_reduced = TSNE(n_components=2, perplexity=30).fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=clusters if not USE_LABELS else labels,
    cmap="tab10",
    s=10,
)
plt.title("Wizualizacja klastrów dokumentów (t-SNE)")
plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ Gotowe!")
import json

output_path = "processed_texts.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        [{"filename": fn, "text": txt} for fn, txt in zip(filenames, texts)],
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"💾 Zapisano przetworzone dane do pliku: {output_path}")
