import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import base64

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIGIT · Neural Recognition",
    page_icon="🔢",
    layout="centered",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Clash+Display:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background: #f5f2eb;
    min-height: 100vh;
}

/* ── Header ── */
.header {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 3px solid #1a1a1a;
    margin-bottom: 2rem;
}
.header-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7rem;
    line-height: 0.85;
    color: #1a1a1a;
    letter-spacing: -0.02em;
}
.header-text { flex: 1; }
.header-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    color: #1a1a1a;
    letter-spacing: 0.05em;
    line-height: 1;
}
.header-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Cards ── */
.card {
    background: #fff;
    border: 2px solid #1a1a1a;
    border-radius: 0;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 4px 4px 0px #1a1a1a;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 1rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}

/* ── Result display ── */
.result-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 9rem;
    line-height: 0.9;
    color: #1a1a1a;
    text-align: center;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    text-align: center;
    margin-top: 0.3rem;
}
.confidence-bar-wrap {
    margin: 0.6rem 0 0.3rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
}
.conf-digit {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: #1a1a1a;
    width: 14px;
    text-align: right;
}
.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: #eee;
    border-radius: 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: #1a1a1a;
    border-radius: 0;
    transition: width 0.5s ease;
}
.conf-bar-fill.top {
    background: #ff3b30;
}
.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #aaa;
    width: 38px;
    text-align: right;
}

/* ── Model stats ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
}
.stat-box {
    flex: 1;
    background: #f5f2eb;
    border: 1px solid #ddd;
    padding: 0.75rem;
    text-align: center;
}
.stat-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #1a1a1a;
    line-height: 1;
}
.stat-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #bbb;
    margin-top: 0.2rem;
}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 2px dashed #ccc !important;
    border-radius: 0 !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #1a1a1a !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    background: #1a1a1a !important;
    color: #f5f2eb !important;
    border: 2px solid #1a1a1a !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.6rem !important;
    box-shadow: 3px 3px 0 #888 !important;
    transition: all 0.1s !important;
}
div[data-testid="stButton"] button:hover {
    box-shadow: 1px 1px 0 #888 !important;
    transform: translate(2px, 2px) !important;
}

/* ── Misc ── */
.tip {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #bbb;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}
.pill {
    display: inline-block;
    background: #1a1a1a;
    color: #f5f2eb;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.15rem 0.6rem;
    margin-right: 0.35rem;
}
section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Model Training ───────────────────────────────────────────────────────────
MODEL_PATH = "/tmp/digit_mlp.pkl"

@st.cache_resource(show_spinner=False)
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, scaler, acc, len(X_train), len(X_test)

with st.spinner("Training neural network…"):
    model, scaler, accuracy, n_train, n_test = train_model()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="header-num">0–9</div>
    <div class="header-text">
        <div class="header-title">DIGIT</div>
        <div class="header-sub">Neural Digit Recognition · MLP Classifier</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Info ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="card">
    <div class="card-label">Model Status</div>
    <div>
        <span class="pill">Ready</span>
        <span class="pill">MLP · 3 Layers</span>
        <span class="pill">sklearn</span>
    </div>
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-val">{accuracy*100:.1f}%</div>
            <div class="stat-key">Test Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{n_train}</div>
            <div class="stat-key">Train Samples</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">256·128·64</div>
            <div class="stat-key">Hidden Layers</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">10</div>
            <div class="stat-key">Classes (0–9)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Image Preprocessing ─────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert any uploaded image → 8×8 grayscale feature vector like sklearn digits."""
    img = img.convert("L")                        # grayscale
    img = ImageOps.invert(img)                    # invert: white digit on black bg
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)                      # tight crop
    img = img.resize((8, 8), Image.LANCZOS)       # 8×8 like sklearn digits
    arr = np.array(img, dtype=np.float64)
    arr = arr / arr.max() * 16.0 if arr.max() > 0 else arr   # scale 0–16
    return arr.flatten()


def predict(img: Image.Image):
    vec = preprocess_image(img)
    vec_scaled = scaler.transform([vec])
    pred = model.predict(vec_scaled)[0]
    proba = model.predict_proba(vec_scaled)[0]
    return int(pred), proba


# ─── Input Section ───────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Input — Upload a handwritten digit image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an image of a handwritten digit (0–9)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="collapsed",
)
st.markdown('<div class="tip">→ Best results: white/light digit on dark background, square crop, single digit per image.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─── Sample digits from sklearn ──────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Or — Try a sample from the test set</div>', unsafe_allow_html=True)

from sklearn.datasets import load_digits as _ld
_digits = _ld()
_X_raw = _digits.data
_y_raw = _digits.target

cols = st.columns(10)
sample_idx = None
for d in range(10):
    idxs = np.where(_y_raw == d)[0]
    pick = idxs[5]  # pick the 6th sample for each digit
    arr8 = _X_raw[pick].reshape(8, 8)
    # render as tiny PIL image
    thumb = Image.fromarray((arr8 / 16.0 * 255).astype(np.uint8)).resize((40, 40), Image.NEAREST)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    with cols[d]:
        if st.button(str(d), key=f"sample_{d}"):
            sample_idx = pick

st.markdown('</div>', unsafe_allow_html=True)

# ─── Run prediction ──────────────────────────────────────────────────────────
prediction = None
confidence = None
img_display = None

if uploaded is not None:
    img = Image.open(uploaded)
    img_display = img
    prediction, confidence = predict(img)

elif sample_idx is not None:
    arr8 = _X_raw[sample_idx].reshape(8, 8)
    pil = Image.fromarray((arr8 / 16.0 * 255).astype(np.uint8))
    img_display = pil
    vec = _X_raw[sample_idx]
    vec_scaled = scaler.transform([vec])
    prediction = int(model.predict(vec_scaled)[0])
    confidence = model.predict_proba(vec_scaled)[0]


# ─── Results ─────────────────────────────────────────────────────────────────
if prediction is not None and confidence is not None:

    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown('<div class="card"><div class="card-label">Input Image</div>', unsafe_allow_html=True)
        st.image(img_display, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="card"><div class="card-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-big">{prediction}</div>
        <div class="result-label">Predicted digit · {confidence[prediction]*100:.1f}% confidence</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Confidence bars ──
    st.markdown('<div class="card"><div class="card-label">Confidence Distribution — All Classes</div>', unsafe_allow_html=True)

    bars_html = '<div class="confidence-bar-wrap">'
    sorted_idxs = np.argsort(confidence)[::-1]
    for i in range(10):
        d = i
        pct = confidence[d] * 100
        is_top = (d == prediction)
        fill_class = "conf-bar-fill top" if is_top else "conf-bar-fill"
        bars_html += f"""
        <div class="conf-row">
            <div class="conf-digit">{d}</div>
            <div class="conf-bar-bg"><div class="{fill_class}" style="width:{pct:.1f}%"></div></div>
            <div class="conf-pct">{pct:.1f}%</div>
        </div>"""
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Placeholder
    st.markdown("""
    <div class="card" style="text-align:center;padding:3rem 1rem;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:5rem;color:#ddd;line-height:1;">?</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#ccc;margin-top:0.5rem;">
            Upload an image or select a sample above
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Architecture explainer ──────────────────────────────────────────────────
with st.expander("How it works"):
    st.markdown("""
**Architecture**  
A Multi-Layer Perceptron (MLP) trained on the scikit-learn `load_digits` dataset — 1,797 samples of 8×8 grayscale images of handwritten digits.

**Pipeline**
1. Input image → grayscale → inverted → tight-cropped → resized to 8×8
2. Pixel values scaled to range 0–16 (matching sklearn digits format)
3. StandardScaler normalisation
4. MLP forward pass → softmax → class probabilities

**Network shape**  
Input (64) → Dense 256 → Dense 128 → Dense 64 → Output (10)  
Activation: ReLU · Optimizer: Adam · Early stopping enabled
    """)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:#ccc;border-top:1px solid #ddd;padding:1.5rem 0 1rem;margin-top:2rem;">
DIGIT · MLP Classifier · scikit-learn · built with Streamlit
</div>
""", unsafe_allow_html=True)
