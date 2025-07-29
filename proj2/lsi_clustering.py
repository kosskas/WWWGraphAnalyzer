import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import spacy

nlp = spacy.load("pl_core_news_sm")


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


def load_documents(folder):
    texts = []
    filenames = []
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            texts.append(preprocess(f.read()))
            filenames.append(fname)
    return texts, filenames


def lsi_clustering(texts, n_topics=100, n_clusters=5):
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
    X_tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_topics)
    X_lsi = svd.fit_transform(X_tfidf)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_lsi)

    score = silhouette_score(X_lsi, labels)
    return labels, X_lsi, score, model
