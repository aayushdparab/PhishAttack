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

ocr_reader = easyocr.Reader(["en"], gpu=False)

def ocr_image(pil_image):
    img_np = np.array(pil_image)
    results = ocr_reader.readtext(img_np, detail=0)
    return " ".join(results)

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
    except:
        text, points, _ = detector.detectAndDecode(arr)
        return [text] if text else []

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "phishing_detector.pkl"
VECT_PATH = BASE_DIR.parent / "models" / "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

def classify_text(text):
    X_text = vectorizer.transform([text]).toarray()
    urls = extract_urls(text)
    X_urls = np.array(url_features(urls)).reshape(1, -1)
    try:
        X = np.hstack([X_text, X_urls])
    except:
        X = X_text
    pred = int(model.predict(X)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X).max())
        except:
            pass
    return pred, prob, urls

st.title("🛡️ PhishGuard AI — Email & Screenshot Phishing Detector")

tab1, tab2 = st.tabs(["📄 Text Email", "🖼️ Screenshot Image"])

with tab1:
    email_input = st.text_area("Paste an email here:", height=250)
    if st.button("Analyze Text Email"):
        if email_input.strip() == "":
            st.error("Please paste an email.")
        else:
            pred, prob, urls = classify_text(email_input)
            label = "🚨 Phishing" if pred == 1 else "✅ Not Phishing"
            st.markdown(f"### Prediction: {label}")
            if prob is not None:
                st.write(f"**Confidence:** {prob:.2f}")
            st.write("### Extracted URLs:")
            for u in urls:
                st.write("-", u)

with tab2:
    uploaded_img = st.file_uploader("Upload a screenshot of an email (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        pil_img = Image.open(uploaded_img).convert("RGB")
        st.image(pil_img, caption="Uploaded Screenshot", use_column_width=True)
        extracted_text = ocr_image(pil_img)
        st.text_area("OCR Extracted Text", extracted_text, height=200)
        qr_data = decode_qr(pil_img)
        if qr_data:
            st.write("### QR / Barcode Detected:")
            for item in qr_data:
                st.write("-", item)
        pred, prob, urls = classify_text(extracted_text)
        label = "🚨 Phishing" if pred == 1 else "✅ Not Phishing"
        st.markdown(f"## Prediction: {label}")
        if prob:
            st.write(f"**Confidence:** {prob:.2f}")
        st.write("### Extracted URLs:")
        for u in urls:
            st.write("-", u)

st.markdown("---")
st.write("Deny the Phish Attacks!!!!")


