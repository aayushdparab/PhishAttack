# preprocess.py
import pickle
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text or "")

def url_features(urls):
    return [
        len(urls),
        int(any("verify" in u.lower() or "login" in u.lower() for u in urls)),
        int(any(re.match(r'https?://\d+\.\d+\.\d+\.\d+', u) for u in urls)),
        len(set(urlparse(u).netloc for u in urls if "://" in u or u.startswith("www."))),
    ]

def preprocess_to_pickle(input_csv="data/phishing_emails.csv", out_pkl="data/preprocessed_data.pkl"):
    df = pd.read_csv(input_csv, encoding="latin1")
    if 'email_content' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'email_content' and 'label' columns.")
    texts = df['email_content'].astype(str).tolist()
    labels = df['label'].astype(int).values

    vect = TfidfVectorizer()
    X_text = vect.fit_transform(texts).toarray()
    X_urls = np.array([url_features(extract_urls(t)) for t in texts])
    X = np.hstack([X_text, X_urls])
    with open(out_pkl, "wb") as f:
        pickle.dump((X, labels, vect), f)
    print(f"Saved preprocessed data -> {out_pkl}")

if __name__ == "__main__":
    preprocess_to_pickle()
