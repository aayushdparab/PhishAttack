# --- pyzbar stub to avoid zbar import errors on Streamlit Cloud ---
import sys, types
_stub = types.ModuleType("pyzbar")
_stub_pyzbar_pyzbar = types.ModuleType("pyzbar.pyzbar")
def _dummy_decode(*args, **kwargs):
    return []  # Always return empty (we use OpenCV QR decode instead)
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

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

# -----------------------------
# URL extraction
# -----------------------------
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

# -----------------------------
# OCR LOADER (Cached)
# -----------------------------
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

ocr_reader = get_ocr_reader()

def ocr_image(pil_image):
    img_np = np.array(pil_image)
    results = ocr_reader.readtext(img_np, detail=0)
    return " ".join(results)

# -----------------------------
# QR Decode via OPENCV ONLY
# -----------------------------
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

# -----------------------------
# MODEL LOADING
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "phishing_detector.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"

model = None
vectorizer = None
loaded_from = None

def try_load_models():
    global model, vectorizer, loaded_from

    possible_paths = [
        MODEL_DIR,
        BASE_DIR / "models",
        Path.cwd() / "models",
        Path.home() / "models"
    ]

    for base in possible_paths:
        mp = base / "phishing_detector.pkl"
        vp = base / "vectorizer.pkl"

        if mp.exists() and vp.exists():
            try:
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                with open(vp, "rb") as f:
                    v = pickle.load(f)
            except Exception:
                continue

            model = m
            vectorizer = v
            loaded_from = str(base)
            return

    model = None
    vectorizer = None
    loaded_from = None

try_load_models()

# If missing model → ask user to upload
if model is None or vectorizer is None:
    st.warning("Model or vectorizer not found. Upload them below.")
    col1, col2 = st.columns(2)
    up_m = col1.file_uploader("Upload phishing_detector.pkl", type=["pkl"])
    up_v = col2.file_uploader("Upload vectorizer.pkl", type=["pkl"])

    if up_m and up_v:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(up_m.getbuffer())
        with open(VECT_PATH, "wb") as f:
            f.write(up_v.getbuffer())
        try_load_models()
        st.success("Models uploaded successfully. Rerun the app.")
    st.stop()

st.success(f"Model loaded from: {loaded_from}")

# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify_text(text):
    if vectorizer is None or model is None:
        return 0, None, []

    try:
        X_text = vectorizer.transform([text]).toarray()
    except Exception:
        X_text = np.zeros((1, 0))

    urls = extract_urls(text)
    X_urls = np.array(url_features(urls)).reshape(1, -1)

    try:
        if X_text.size and X_urls.size:
            X = np.hstack([X_text, X_urls])
        elif X_text.size:
            X = X_text
        else:
            X = X_urls
    except Exception:
        X = X_text if X_text.size else X_urls

    try:
        pred = int(model.predict(X)[0])
    except Exception:
        pred = 0

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X).max())
        except Exception:
            prob = None

    return pred, prob, urls

# -----------------------------
# UI
# -----------------------------
st.title("🛡️ PhishGuard AI — Email & Screenshot Phishing Detector")

tab1, tab2 = st.tabs(["📄 Text Email", "🖼️ Screenshot Image"])

# TEXT EMAIL TAB
with tab1:
    email_input = st.text_area("Paste an email here:", height=250)
    if st.button("Analyze Text Email"):
        if not email_input.strip():
            st.error("Please paste an email.")
        else:
            pred, prob, urls = classify_text(email_input)
            label = "🚨 Phishing" if pred == 1 else "✅ Not Phishing"
            st.markdown(f"### Prediction: {label}")
            if prob is not None:
                st.write(f"**Confidence:** {prob:.2f}")
            if urls:
                st.write("### Extracted URLs:")
                for u in urls:
                    st.write("-", u)

# SCREENSHOT TAB
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

        pred, prob, urls = classify_text(extracted_text)
        label = "🚨 Phishing" if pred == 1 else "✅ Not Phishing"
        st.markdown(f"## Prediction: {label}")

        if prob is not None:
            st.write(f"**Confidence:** {prob:.2f}")

        if urls:
            st.write("### Extracted URLs:")
            for u in urls:
                st.write("-", u)

st.markdown("---")
st.write("Deny the Phish Attacks!!!!")




