import pickle
import numpy as np
import pandas as pd
import os
import re
from urllib.parse import urlparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

def train_model(raw_data_file="data/phishing_emails.csv",
                model_file="models/phishing_detector.pkl",
                vectorizer_file="models/vectorizer.pkl",
                feedback_file="data/feedback.csv",
                test_size=0.2,
                random_state=42):
    """
    Expects raw_data_file to be a CSV with columns:
      - email_content (text)
      - label (0 for not phishing, 1 for phishing)
    """

    raw_path = Path(raw_data_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {raw_data_file}")

    df = pd.read_csv(raw_path, encoding="latin1")
    if 'email_content' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'email_content' and 'label' columns.")

    texts = df['email_content'].astype(str).tolist()
    labels = df['label'].astype(int).values

    vectorizer = TfidfVectorizer(max_features=None)
    X_text = vectorizer.fit_transform(texts).toarray()

    url_feat_list = [url_features(extract_urls(t)) for t in texts]
    X_urls = np.array(url_feat_list)

    X = np.hstack([X_text, X_urls])

    feedback_path = Path(feedback_file)
    if feedback_path.exists():
        try:
            df_fb = pd.read_csv(feedback_path)
            df_fb = df_fb.dropna(subset=['email_content', 'correct_label'])
            if not df_fb.empty:
                fb_texts = df_fb['email_content'].astype(str).tolist()
                fb_labels = df_fb['correct_label'].astype(int).values
                X_text_fb = vectorizer.transform(fb_texts).toarray()
                X_urls_fb = np.array([url_features(extract_urls(t)) for t in fb_texts])
                X_fb = np.hstack([X_text_fb, X_urls_fb])
                X = np.vstack([X, X_fb])
                labels = np.concatenate([labels, fb_labels])
                print(f"Added {len(fb_labels)} feedback samples to training data.")
        except Exception as e:
            print("Warning: could not read feedback file:", e)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state, stratify=labels)

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_file), exist_ok=True)

    with open(model_file, "wb") as mf:
        pickle.dump(clf, mf)
    with open(vectorizer_file, "wb") as vf:
        pickle.dump(vectorizer, vf)

    print(f"Saved model -> {model_file}")
    print(f"Saved vectorizer -> {vectorizer_file}")
    print(f"Model expects {getattr(clf, 'n_features_in_', None)} features (TEXT features + 4 URL features).")

if __name__ == "__main__":
    train_model()






