# --- pyzbar stub to avoid zbar import errors on Streamlit Cloud ---
import sys, types
_stub = types.ModuleType("pyzbar")
_stub_pyzbar_pyzbar = types.ModuleType("pyzbar.pyzbar")
def _dummy_decode(*args, **kwargs):
    return []
setattr(_stub_pyzbar_pyzbar, "decode", _dummy_decode)
sys.modules["pyzbar"] = _stub
sys.modules["pyzbar.pyzbar"] = _stub_pyzbar_pyzbar
# --- END STUB ---

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

# ------ Decision thresholds (tunable) ------
PHISH_THRESHOLD = 0.60     # p(phish) >= this -> phishing (high confidence)
NOT_PHISH_THRESHOLD = 0.40 # p(phish) <= this -> not phishing (high confidence)
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

def decide_label_from_prob(prob_phish):
    """
    Decide label string + certainty based on probability of class 1 (phishing).
    Returns (label_text, certainty_tag)
    """
    if prob_phish is None:
        return ("Suspect / Manual Review", "low_confidence")
    if prob_phish >= PHISH_THRESHOLD:
        return ("Phishing", "high_confidence")
    if prob_phish <= NOT_PHISH_THRESHOLD:
        return ("Not Phishing", "high_confidence")
    if UNCERTAIN_LOW <= prob_phish <= UNCERTAIN_HIGH:
        return ("Suspect / Manual Review", "uncertain")
    return ("Suspect / Manual Review", "low_confidence")

# ---- URL helpers ----
def extract_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.findall(text or "")

def url_features(urls):
    return [
        len(urls),
        int(any("verify" in u.lower() or "login" in u.lower() for u in urls)),
        int(any(re.match(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls)),
        len(set(urlparse(u).netloc for u in urls if "://" in u or u.startswith("www.")))
    ]

# ---- OCR reader (cached) ----
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)
ocr_reader = get_ocr_reader()

def ocr_image(pil_image):
    img_np = np.array(pil_image)
    results = ocr_reader.readtext(img_np, detail=0)
    return " ".join(results)

# ---- QR decode (OpenCV) ----
def decode_qr(pil_img):
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
        try:
            text, points, _ = detector.detectAndDecode(arr)
            return [text] if text else []
        except Exception:
            return []

# ---- model loader ----
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

# ---- feedback saver ----
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

# ---- pad/truncate helper ----
def pad_or_truncate(arr, target_dim):
    if target_dim is None:
        return arr
    cur = arr.shape[1]
    if cur == target_dim:
        return arr
    if cur < target_dim:
        pad = np.zeros((1, target_dim - cur))
        return np.hstack([arr, pad])
    else:
        return arr[:, :target_dim]

# ---- classify_text: returns pred_label (by prob>=0.5), prob_phish, urls ----
def classify_text(text):
    if vectorizer is None or model is None:
        return 0, None, []
    try:
        X_text = vectorizer.transform([text]).toarray()
    except Exception:
        X_text = np.zeros((1, 0))
    urls = extract_urls(text)
    X_urls = np.array(url_features(urls)).reshape(1, -1)

    # metadata dims
    if metadata and isinstance(metadata, dict):
        text_dim = metadata.get("text_features") or metadata.get("tfidf_max_features") or X_text.shape[1]
        url_dim = metadata.get("url_features") or X_urls.shape[1]
    else:
        text_dim = X_text.shape[1]
        url_dim = X_urls.shape[1]

    X_text = pad_or_truncate(X_text, text_dim)
    X_urls = pad_or_truncate(X_urls, url_dim)

    try:
        X = np.hstack([X_text, X_urls])
    except Exception:
        X = X_text if X_text.size else X_urls

    prob_phish = None
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # binary classifier: column 1 is prob of class 1
            if p.shape[1] == 2:
                prob_phish = float(p[0, 1])
            else:
                # fallback: take max or try index for label 1
                prob_phish = float(p.max())
        else:
            prob_phish = None
    except Exception:
        prob_phish = None

    # determine pred_label consistently from prob_phish if available, else model.predict
    if prob_phish is not None:
        pred_label = 1 if prob_phish >= 0.5 else 0
    else:
        try:
            pred_label = int(model.predict(X)[0])
        except Exception:
            pred_label = 0

    return pred_label, prob_phish, urls

# ---- UI ----
st.title("🛡️ PhishGuard AI — Email & Screenshot Phishing Detector")
tab1, tab2 = st.tabs(["Upload Text", "Upload Image"])

with tab1:
    email_input = st.text_area("Paste an email here:", height=250)
    if st.button("Analyze Text Email"):
        if not email_input.strip():
            st.error("Please paste an email.")
        else:
            pred_label, prob_phish, urls = classify_text(email_input)
            label_text, certainty = decide_label_from_prob(prob_phish)
            emoji = "🚨" if label_text.startswith("Phishing") or label_text.startswith("Suspect") else "✅"
            st.markdown(f"### Prediction: {emoji} {label_text}")
            if prob_phish is not None:
                st.write(f"**Model confidence (p(phish)):** {prob_phish:.2f}")
            if certainty == "uncertain":
                st.warning("⚠️ Low confidence — please review manually or provide feedback.")
            elif certainty == "low_confidence":
                st.info("ℹ️ Low-confidence prediction (model may be unsure). Consider manual verification.")
            if urls:
                st.write("### Extracted URLs:")
                for u in urls:
                    st.write("-", u)
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
        pred_label, prob_phish, urls = classify_text(extracted_text)
        label_text, certainty = decide_label_from_prob(prob_phish)
        emoji = "🚨" if label_text.startswith("Phishing") or label_text.startswith("Suspect") else "✅"
        st.markdown(f"## Prediction: {emoji} {label_text}")
        if prob_phish is not None:
            st.write(f"**Model confidence (p(phish)):** {prob_phish:.2f}")
        if certainty == "uncertain":
            st.warning("⚠️ Low confidence — OCR may be noisy. Please verify the extracted text above and/or submit feedback.")
        elif certainty == "low_confidence":
            st.info("ℹ️ Low-confidence prediction (model may be unsure). Consider manual verification.")
        if urls:
            st.write("### Extracted URLs:")
            for u in urls:
                st.write("-", u)
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

