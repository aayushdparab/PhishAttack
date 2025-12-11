import pickle
import os
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from pathlib import Path

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

def predict_email(model_file="models/phishing_detector.pkl", vectorizer_file="models/vectorizer.pkl", email_text=None, feedback_file="data/feedback.csv"):
    if email_text is None:
        raise ValueError("email_text must be provided")

    model_path = Path(model_file)
    vect_path = Path(vectorizer_file)
    if not model_path.exists() or not vect_path.exists():
        raise FileNotFoundError("Model or vectorizer pickle not found. Run training first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vect_path, "rb") as f:
        vectorizer = pickle.load(f)

    X_text = vectorizer.transform([email_text]).toarray()
    X_urls = np.array([url_features(extract_urls(email_text))])
    try:
        X = np.hstack([X_text, X_urls])
    except Exception:
        # if model was saved without URL features (unlikely after using train.py), fallback
        X = X_text

    pred = model.predict(X)[0]
    label = "Phishing" if pred == 1 else "Not Phishing"

    # Append to feedback log with no correct_label (can be corrected later)
    row = {"email_content": email_text, "predicted_label": int(pred), "correct_label": None}
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(feedback_file, index=False)
    return label

if __name__ == "__main__":
    text = input("Enter email text:\n")
    print(predict_email(email_text=text))