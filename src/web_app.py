# app.py
import streamlit as st
import pickle
import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sklearn

# ------------------------------
# Config & small CSS
# ------------------------------
st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.card { background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
.big-title { font-size:22px; font-weight:700; margin-bottom:6px; }
.muted { color:#6b7280; font-size:13px; }
.warning { color:#9a3412; font-weight:700; }
</style>""", unsafe_allow_html=True)

# ------------------------------
# Utilities
# ------------------------------
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

# ------------------------------
# Load model + vectorizer
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"  # project_root/models
MODEL_PATH = MODEL_DIR / "phishing_detector.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"

model = None
vectorizer = None

if MODEL_PATH.exists() and VECT_PATH.exists():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECT_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        st.success(f"Loaded model and vectorizer from {MODEL_DIR}")
    except Exception as e:
        st.error(f"Error loading pickles: {e}")
else:
    st.warning("Model or vectorizer not found in models/ folder. Upload them below.")
    m_up = st.file_uploader("Upload phishing_detector.pkl", type=["pkl"])
    v_up = st.file_uploader("Upload vectorizer.pkl", type=["pkl"])
    if m_up and v_up:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(m_up.getbuffer())
        with open(VECT_PATH, "wb") as f:
            f.write(v_up.getbuffer())
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECT_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        st.success("Uploaded and loaded model & vectorizer.")

if model is None or vectorizer is None:
    st.stop()

# ------------------------------
# Session state & helpers
# ------------------------------
if "email_text" not in st.session_state:
    st.session_state.email_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None

def get_vectorizer_feature_count():
    try:
        return vectorizer.transform(["test"]).toarray().shape[1]
    except Exception:
        return None

def get_model_expected():
    return getattr(model, "n_features_in_", None)

def make_feature_vector(text):
    X_text = vectorizer.transform([text]).toarray()
    X_urls = np.array([url_features(extract_urls(text))])
    # combine
    X = np.hstack([X_text, X_urls])
    return X, X_text.shape[1], X_urls.shape[1]

def safe_predict(text):
    X, t_cnt, u_cnt = None, None, None
    try:
        X, t_cnt, u_cnt = make_feature_vector(text)
    except Exception as e:
        raise RuntimeError(f"Vectorizer transform failed: {e}")

    expected = get_model_expected()
    combined = X.shape[1]

    # diagnostics if mismatch
    if expected is not None and combined != expected:
        raise RuntimeError(
            f"Feature mismatch: combined features={combined} (text={t_cnt} + url={u_cnt}) "
            f"but model expects {expected}. Make sure model & vectorizer are from the same training run."
        )

    # final prediction
    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X).max())
        except Exception:
            prob = None
    return int(pred), prob, extract_urls(text)

# ------------------------------
# UI
# ------------------------------
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="big-title">PhishGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Paste an email and press Classify</div>', unsafe_allow_html=True)
    st.markdown("---")
    show_conf = st.checkbox("Show confidence", True)
    show_feedback = st.checkbox("Show feedback dashboard", False)
    st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Email text")
    email_text = st.text_area("", height=240, key="main_email_text", value=st.session_state.email_text)
    if st.button("Classify"):
        if not email_text.strip():
            st.warning("Enter email text.")
        else:
            try:
                pred, prob, urls = safe_predict(email_text)
            except Exception as e:
                st.error(str(e))
                st.info({
                    "scikit-learn": sklearn.__version__,
                    "model_expected_features": get_model_expected(),
                    "vectorizer_text_features": get_vectorizer_feature_count(),
                    "note": "If these don't match, re-run training with train.py so model and vectorizer align."
                })
            else:
                st.session_state.prediction = pred
                st.session_state._last_prob = prob
                st.session_state._last_urls = urls
                st.session_state.email_text = email_text
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Result")
    if st.session_state.get("prediction") is not None:
        pred = st.session_state.prediction
        prob = st.session_state._last_prob
        urls = st.session_state._last_urls or extract_urls(st.session_state.email_text)
        label = "Phishing 🚨" if pred == 1 else "Not Phishing ✅"
        st.markdown(f"#### Prediction: {label}")
        if show_conf and prob is not None:
            st.markdown(f"**Confidence:** {prob:.2f}")

        if urls:
            st.markdown("**URLs found:**")
            for u in urls:
                st.markdown(f"- {u}")

        st.markdown("---")
        st.markdown("#### Feedback")
        with st.form("fb"):
            c1, c2, c3 = st.columns(3)
            ok = c1.form_submit_button("Correct")
            wrong_phish = c2.form_submit_button("Incorrect: Phishing")
            wrong_not = c3.form_submit_button("Incorrect: Not Phishing")
            if ok or wrong_phish or wrong_not:
                if ok:
                    correct = pred
                elif wrong_phish:
                    correct = 1
                else:
                    correct = 0
                fb_dir = BASE_DIR / "data"
                fb_dir.mkdir(exist_ok=True)
                fb_file = fb_dir / "feedback.csv"
                row = {"email_content": st.session_state.email_text, "predicted_label": int(pred), "correct_label": int(correct)}
                df_row = pd.DataFrame([row])
                if fb_file.exists():
                    df_row.to_csv(fb_file, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(fb_file, index=False)
                st.success("Feedback saved. Consider re-training to incorporate feedback.")
                st.session_state.prediction = None
    else:
        st.markdown("No prediction yet - paste email and click **Classify**.")
    st.markdown("</div>", unsafe_allow_html=True)

if show_feedback:
    st.markdown("## Feedback (admin)")
    fb_file = BASE_DIR / "data" / "feedback.csv"
    if fb_file.exists():
        st.dataframe(pd.read_csv(fb_file))
    else:
        st.info("No feedback yet.")

