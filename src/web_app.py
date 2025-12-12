import sys, types
_stub = types.ModuleType("pyzbar")
_stub_pyzbar_pyzbar = types.ModuleType("pyzbar.pyzbar")
def _dummy_decode(*args, **kwargs):
    return []
setattr(_stub_pyzbar_pyzbar, "decode", _dummy_decode)
sys.modules["pyzbar"] = _stub
sys.modules["pyzbar.pyzbar"] = _stub_pyzbar_pyzbar

import streamlit as st
import pickle
import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import easyocr
import cv2
import csv
from datetime import datetime

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

PHISH_THRESHOLD = 0.60
NOT_PHISH_THRESHOLD = 0.30
UNCERTAIN_LOW = NOT_PHISH_THRESHOLD
UNCERTAIN_HIGH = PHISH_THRESHOLD
MODEL_WEIGHT = 0.80
HEUR_WEIGHT = 1.0 - MODEL_WEIGHT

def extract_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.findall(text or "")

def url_features(urls):
    return [
        len(urls),
        int(any("verify" in u.lower() or "login" in u.lower() or "account" in u.lower() for u in urls)),
        int(any(re.match(r'https?://\d+\.\d+\.\d+\.\d+', u) for u in urls)),
        len(set(urlparse(u).netloc for u in urls if "://" in u or u.startswith("www.")))
    ]

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)
ocr_reader = get_ocr_reader()

def ocr_image(pil_image):
    try:
        img_np = np.array(pil_image)
        results = ocr_reader.readtext(img_np, detail=0)
        return " ".join(results)
    except Exception:
        return ""

def decode_qr(pil_img):
    try:
        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        try:
            decoded_texts, points, _ = detector.detectAndDecodeMulti(arr)
            results = []
            if isinstance(decoded_texts, (list, tuple)) and decoded_texts:
                for t in decoded_texts:
                    if t:
                        results.append(t)
            elif isinstance(decoded_texts, str) and decoded_texts:
                results.append(decoded_texts)
            return results
        except Exception:
            text, points, _ = detector.detectAndDecode(arr)
            return [text] if text else []
    except Exception:
        return []

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "phishing_detector.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"
META_PATH = MODEL_DIR / "metadata.pkl"

model = None
vectorizer = None
metadata = None
loaded_from = None

def try_load_models():
    global model, vectorizer, metadata, loaded_from
    possible = [
        MODEL_DIR,
        BASE_DIR / "models",
        Path.cwd() / "models",
        Path.home() / "models"
    ]
    for base in possible:
        mp = base / "phishing_detector.pkl"
        vp = base / "vectorizer.pkl"
        mep = base / "metadata.pkl"
        if mp.exists() and vp.exists():
            try:
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                with open(vp, "rb") as f:
                    v = pickle.load(f)
                meta = None
                if mep.exists():
                    try:
                        with open(mep, "rb") as mf:
                            meta = pickle.load(mf)
                    except Exception:
                        meta = None
            except Exception:
                continue
            model = m
            vectorizer = v
            metadata = meta
            loaded_from = str(base)
            return
    model = None
    vectorizer = None
    metadata = None
    loaded_from = None

try_load_models()

if model is None or vectorizer is None:
    st.warning("Model or vectorizer not found. Upload them below or put them in `models/` at repo root.")
    col1, col2 = st.columns(2)
    up_m = col1.file_uploader("Upload phishing_detector.pkl", type=["pkl"])
    up_v = col2.file_uploader("Upload vectorizer.pkl", type=["pkl"])
    up_meta = st.file_uploader("Optional: upload metadata.pkl", type=["pkl"])
    if up_m and up_v:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(up_m.getbuffer())
        with open(VECT_PATH, "wb") as f:
            f.write(up_v.getbuffer())
        if up_meta:
            with open(META_PATH, "wb") as f:
                f.write(up_meta.getbuffer())
        try_load_models()
        if model is not None and vectorizer is not None:
            st.success("Models uploaded and loaded successfully. Please re-run analysis.")
    st.stop()
else:
    st.success(f"Model loaded from: {loaded_from}")

def save_feedback(email_text, predicted_label, correct_label):
    data_dir = BASE_DIR.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = data_dir / "feedback.csv"
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "email_content": str(email_text).replace("\n", " ").strip(),
        "predicted_label": int(predicted_label),
        "correct_label": int(correct_label)
    }
    header = ["timestamp", "email_content", "predicted_label", "correct_label"]
    write_header = not feedback_file.exists()
    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def pad_or_truncate(arr, target_dim):
    if target_dim is None or arr is None:
        return arr
    if arr.size == 0:
        return np.zeros((1, target_dim))
    cur = arr.shape[1]
    if cur == target_dim:
        return arr
    if cur < target_dim:
        pad = np.zeros((1, target_dim - cur))
        return np.hstack([arr, pad])
    else:
        return arr[:, :target_dim]

def _model_phish_probability(model_obj, X):
    if model_obj is None:
        return None
    if not hasattr(model_obj, "predict_proba"):
        return None
    try:
        p = model_obj.predict_proba(X)
        classes = list(getattr(model_obj, "classes_", []))
        if 1 in classes:
            idx = classes.index(1)
            return float(p[0, idx])
        elif p.shape[1] == 2:
            return float(p[0, 1])
        else:
            return float(p[0].max())
    except Exception:
        return None

def _heuristic_score(text, urls):
    score = 0.0
    text_lower = (text or "").lower()
    strong_phrases = [
        "verify your account", "urgent action", "confirm your", "billing problem",
        "click here to login", "account suspended", "update your payment",
        "view confidential feedback", "view confidential", "click below", "click here",
        "view confidential link", "view confidential message", "confirm your account"
    ]
    if any(p in text_lower for p in strong_phrases):
        score += 0.40

    if urls:
        suspicious_keywords = ["verify", "login", "account-update", "secure", "password", "confirm"]
        if any(any(k in u.lower() for k in suspicious_keywords) for u in urls):
            score += 0.35
        if any(re.match(r'https?://\d+\.\d+\.\d+\.\d+', u) for u in urls):
            score += 0.25
        domains = set(urlparse(u).netloc for u in urls if u and ("://" in u or u.startswith("www.")))
        if len(domains) >= 3:
            score += 0.20
        if len(urls) > 2:
            score += 0.10

    if re.search(r"\b(view|click)\b.*\b(confidential|feedback|here|below)\b", text_lower):
        score += 0.25

    return min(score, 0.99)

def classify_text(text):
    if vectorizer is None or model is None:
        return 0, None, [], 0.0, None
    try:
        X_text = vectorizer.transform([text]).toarray()
    except Exception:
        X_text = np.zeros((1, 0))
    urls = extract_urls(text)
    X_urls = np.array(url_features(urls)).reshape(1, -1)
    if metadata and isinstance(metadata, dict):
        text_dim = metadata.get("text_features", X_text.shape[1])
        url_dim = metadata.get("url_features", X_urls.shape[1])
    else:
        text_dim = X_text.shape[1]
        url_dim = X_urls.shape[1]
    X_text = pad_or_truncate(X_text, text_dim)
    X_urls = pad_or_truncate(X_urls, url_dim)
    try:
        if X_text is not None and X_urls is not None and X_text.size and X_urls.size:
            X = np.hstack([X_text, X_urls])
        elif X_text is not None and X_text.size:
            X = X_text
        else:
            X = X_urls
    except Exception:
        X = X_text if (X_text is not None and X_text.size) else X_urls
    model_prob = _model_phish_probability(model, X)
    if model_prob is None:
        try:
            p_only = int(model.predict(X)[0])
            model_prob = 0.99 if p_only == 1 else 0.01
        except Exception:
            model_prob = None
    heur = _heuristic_score(text, urls)
    if heur >= 0.30:
        combined = max(0.95, heur)
    else:
        if model_prob is None:
            combined = heur
        else:
            combined = float(max(0.0, min(1.0, model_prob * MODEL_WEIGHT + heur * HEUR_WEIGHT)))
    if model_prob is not None and model_prob <= 0.05 and heur <= 0.05:
        combined = min(combined, 0.05)
    pred_label = 1 if combined >= 0.5 else 0
    return pred_label, model_prob, urls, float(heur), float(combined)

def decide_label_from_combined(combined_score):
    if combined_score is None:
        return ("Suspect / Manual Review", "low_confidence")
    if combined_score >= PHISH_THRESHOLD:
        return ("Phishing", "high_confidence")
    if combined_score <= NOT_PHISH_THRESHOLD:
        return ("Not Phishing", "high_confidence")
    if UNCERTAIN_LOW <= combined_score <= UNCERTAIN_HIGH:
        return ("Suspect / Manual Review", "uncertain")
    return ("Suspect / Manual Review", "low_confidence")

with st.sidebar:
    st.markdown("## PhishGuard Controls")
    show_conf = st.checkbox("Show confidence", value=True)
    show_debug = st.checkbox("Show debug info (temp)", value=False)
    st.markdown("---")

st.title("🛡️ PhishGuard AI — Email & Screenshot Phishing Detector")
tab1, tab2 = st.tabs(["Upload Text", "Upload Image"])

with tab1:
    email_input = st.text_area("Paste an email here:", height=250)
    if st.button("Analyze Text Email"):
        if not email_input.strip():
            st.error("Please paste an email.")
        else:
            pred_label, model_prob, urls, heur, combined = classify_text(email_input)
            label_text, certainty = decide_label_from_combined(combined)
            emoji = "🚨" if label_text.startswith("Phishing") or label_text.startswith("Suspect") else "✅"
            st.markdown(f"### Prediction: {emoji} {label_text}")
            if show_conf:
                st.write(f"**Model confidence (p(phish))**: {model_prob:.2f}" if model_prob is not None else "**Model confidence (p(phish))**: N/A")
                st.write(f"**Heuristic score:** {heur:.2f}")
                st.write(f"**Combined score:** {combined:.2f}")
            if certainty == "uncertain":
                st.warning("⚠️ Low confidence — please review manually or provide feedback.")
            elif certainty == "low_confidence":
                st.info("ℹ️ Low-confidence prediction (model may be unsure). Consider manual verification.")
            if urls:
                st.write("### Extracted URLs:")
                for u in urls:
                    st.write("-", u)
            if show_debug:
                with st.expander("🔍 Debug internals (details)"):
                    st.write("Model prob:", model_prob)
                    st.write("Heuristic:", heur)
                    st.write("Combined:", combined)
                    st.write("Extracted URLs:", urls)
            st.markdown("#### Was this prediction correct?")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Correct (Text)"):
                    save_feedback(email_input, pred_label, pred_label)
                    st.success("Thanks — feedback saved.")
            with col2:
                if st.button("🚨 Incorrect: This WAS phishing (Text)"):
                    save_feedback(email_input, pred_label, 1)
                    st.success("Thanks — feedback saved.")
            with col3:
                if st.button("✅ Incorrect: This WAS NOT phishing (Text)"):
                    save_feedback(email_input, pred_label, 0)
                    st.success("Thanks — feedback saved.")

with tab2:
    uploaded_img = st.file_uploader("Upload a screenshot (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        pil_img = Image.open(uploaded_img).convert("RGB")
        st.image(pil_img, caption="Uploaded Screenshot", use_column_width=True)
        extracted_text = ocr_image(pil_img)
        st.text_area("OCR Extracted Text", extracted_text, height=200)
        qr_data = decode_qr(pil_img)
        if qr_data:
            st.write("### QR / Barcode Detected:")
            for x in qr_data:
                st.write("-", x)
        pred_label, model_prob, urls, heur, combined = classify_text(extracted_text)
        label_text, certainty = decide_label_from_combined(combined)
        emoji = "🚨" if label_text.startswith("Phishing") or label_text.startswith("Suspect") else "✅"
        st.markdown(f"## Prediction: {emoji} {label_text}")
        if show_conf:
            st.write(f"**Model confidence (p(phish))**: {model_prob:.2f}" if model_prob is not None else "**Model confidence (p(phish))**: N/A")
            st.write(f"**Heuristic score:** {heur:.2f}")
            st.write(f"**Combined score:** {combined:.2f}")
        if certainty == "uncertain":
            st.warning("⚠️ Low confidence — OCR may be noisy. Please verify the extracted text above and/or submit feedback.")
        elif certainty == "low_confidence":
            st.info("ℹ️ Low-confidence prediction (model may be unsure). Consider manual verification.")
        if urls:
            st.write("### Extracted URLs:")
            for u in urls:
                st.write("-", u)
        if show_debug:
            with st.expander("🔍 Debug internals (details)", expanded=True):
                st.write("Model prob:", model_prob)
                st.write("Heuristic:", heur)
                st.write("Combined:", combined)
                st.write("Extracted URLs:", urls)
        st.markdown("#### Was this prediction correct?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Correct (Image)"):
                save_feedback(extracted_text, pred_label, pred_label)
                st.success("Thanks — feedback saved.")
        with col2:
            if st.button("🚨 Incorrect: This WAS phishing (Image)"):
                save_feedback(extracted_text, pred_label, 1)
                st.success("Thanks — feedback saved.")
        with col3:
            if st.button("✅ Incorrect: This WAS NOT phishing (Image)"):
                save_feedback(extracted_text, pred_label, 0)
                st.success("Thanks — feedback saved.")

st.markdown("---")
st.write("Deny the Phish Attacks!!!!")




