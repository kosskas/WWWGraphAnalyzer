from tqdm import tqdm
import os
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load("pl_core_news_sm")


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def load_documents_from_html(folder):
    texts = []
    filenames = []

    html_files = [f for f in os.listdir(folder)]
    html_files = html_files[:100]
    for fname in tqdm(html_files, desc="Przetwarzanie dokumentów HTML"):
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            raw_html = f.read()
            plain_text = extract_text_from_html(raw_html)
            preprocessed = preprocess(plain_text)
            texts.append(preprocessed)
            filenames.append(fname)

    return texts, filenames
