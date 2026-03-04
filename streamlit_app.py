import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import time
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── MODEL DEFINITION ───────────────────────────────────────────────────────────
# Define the same CNN architecture used during training on Kaggle
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# ─── DOWNLOAD MODEL FROM KAGGLE ─────────────────────────────────────────────────
MODEL_PATH = "/tmp/best_model.pth"

def download_model_from_kaggle():
    """
    Download best_model.pth from a Kaggle Model instance.
    Source: taitruong256/hrnet-model/pytorch/default/1/best_model.pth
    """
    if os.path.exists(MODEL_PATH):
        return None  # already cached in /tmp for this session

    try:
        import json

        # ── Write Kaggle credentials from Streamlit secrets ──
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        creds_path = os.path.join(kaggle_dir, "kaggle.json")
        if not os.path.exists(creds_path):
            creds = {
                "username": st.secrets["KAGGLE_USERNAME"],
                "key":      st.secrets["KAGGLE_KEY"],
            }
            with open(creds_path, "w") as f:
                json.dump(creds, f)
            os.chmod(creds_path, 0o600)

        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()

        # ── Download the specific model instance ──
        # Path: taitruong256/hrnet-model/pytorch/default/1
        api.model_instance_version_download(
            owner_slug    = "taitruong256",
            model_slug    = "hrnet-model",
            framework     = "pytorch",
            instance_slug = "default",
            version_number= 1,
            path          = "/tmp",
            quiet         = False,
            untar         = True,
        )

        # The download extracts into /tmp — find best_model.pth
        for root, _, files in os.walk("/tmp"):
            for fname in files:
                if fname == "best_model.pth":
                    src = os.path.join(root, fname)
                    if src != MODEL_PATH:
                        os.rename(src, MODEL_PATH)
                    return None

        return "⚠️ `best_model.pth` not found after extracting the Kaggle model archive."

    except KeyError as e:
        return f"⚠️ Missing Streamlit secret: {e}. Add KAGGLE_USERNAME and KAGGLE_KEY in app Secrets."
    except Exception as e:
        return f"⚠️ Failed to download model from Kaggle: {e}"


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    err = download_model_from_kaggle()
    if err:
        return None, err

    model = DigitCNN()
    try:
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"⚠️ Error loading model weights: {e}"


# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
def preprocess_image(img_array: np.ndarray) -> torch.Tensor:
    """Convert canvas RGBA array → 28×28 normalised tensor."""
    img = Image.fromarray(img_array.astype("uint8"), "RGBA")
    img = img.convert("L")                        # grayscale
    img = ImageOps.invert(img)                    # white bg → black bg
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((20, 20), Image.LANCZOS)
    canvas28 = Image.new("L", (28, 28), 0)
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    canvas28.paste(img, offset)
    arr = np.array(canvas28, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081              # MNIST normalisation
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
    return pred, conf, probs


# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0d0d0f;
    --surface: #16161a;
    --border: #2a2a32;
    --accent: #e8ff47;
    --accent2: #ff6b6b;
    --text: #f0f0f0;
    --muted: #6b6b7a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

.main .block-container {
    max-width: 720px;
    padding: 2rem 2rem 4rem;
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.header-badge {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
}
.header-title span { color: var(--accent); }
.header-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.6rem;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Prediction box ── */
.pred-box {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: #0d0d0f;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
.pred-digit {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    min-width: 70px;
    text-align: center;
}
.pred-meta { flex: 1; }
.pred-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
}
.pred-conf {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}
.conf-bar-wrap {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), #b8d900);
    transition: width 0.4s ease;
}

/* ── Prob bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.prob-digit-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    width: 16px;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    width: 42px;
    text-align: right;
}

/* ── Instructions ── */
.instructions {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.step {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    flex: 1;
    min-width: 140px;
}
.step-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    background: var(--accent);
    color: #000;
    width: 18px;
    height: 18px;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-text {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.4;
}

/* ── Button ── */
div.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.65rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* ── Canvas wrapper ── */
.canvas-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    justify-content: center;
}

/* ── Error ── */
.error-box {
    background: #1a0f0f;
    border: 1px solid var(--accent2);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: var(--accent2);
    font-size: 0.9rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 3rem;
    font-family: 'Space Mono', monospace;
}
.footer span { color: var(--accent); }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-badge">MNIST · PyTorch · CNN</div>
    <h1 class="header-title">Digit<span>.</span>AI</h1>
    <p class="header-sub">Draw a digit and let the neural network do the rest</p>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
with st.spinner("🔗 Fetching model from Kaggle… (first load only, ~10s)"):
    model, error = load_model()
if error:
    st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
    st.stop()

# ─── INSTRUCTIONS ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-title">How to use</div>
    <div class="instructions">
        <div class="step"><div class="step-num">1</div><div class="step-text">Draw any digit (0–9) in the canvas below</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text">Click <strong>Predict</strong> to run inference</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text">See the prediction and confidence breakdown</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text">Hit <strong>Clear</strong> to draw again</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── CANVAS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Draw here</div>', unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.markdown('</div>', unsafe_allow_html=True)

# ─── BUTTONS ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    predict_btn = st.button("⚡  Predict Digit", key="predict")
with col2:
    clear_btn = st.button("✕  Clear", key="clear")

if clear_btn:
    st.rerun()

# ─── PREDICTION ─────────────────────────────────────────────────────────────────
if predict_btn:
    if canvas_result.image_data is None or canvas_result.image_data.sum() == 0:
        st.warning("Draw a digit first!")
    else:
        with st.spinner("Running inference…"):
            tensor = preprocess_image(canvas_result.image_data)
            pred, conf, probs = predict(model, tensor)
            time.sleep(0.3)   # small delay for UX feel

        # ── Main prediction card ──
        st.markdown(f"""
        <div class="card" style="margin-top:1.2rem">
            <div class="card-title">Prediction</div>
            <div class="pred-box">
                <div class="pred-digit">{pred}</div>
                <div class="pred-meta">
                    <div class="pred-label">Confidence</div>
                    <div class="pred-conf">{conf*100:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{conf*100:.1f}%"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability breakdown ──
        st.markdown('<div class="card"><div class="card-title">All class probabilities</div>', unsafe_allow_html=True)

        COLORS = [
            "#e8ff47","#a8ff78","#78ffd6","#48c6ef","#6f86d6",
            "#f093fb","#f5576c","#fd746c","#ff9a44","#ffd452"
        ]

        bars_html = ""
        sorted_idx = np.argsort(probs)[::-1]
        for i in range(10):
            d = i
            p = probs[d]
            is_top = d == pred
            color = COLORS[d]
            width_pct = p * 100
            border = f"border: 1px solid {color}22;" if is_top else ""
            bars_html += f"""
            <div class="prob-row" style="{'background:#ffffff08;border-radius:6px;padding:2px 4px;' if is_top else ''}">
                <div class="prob-digit-label" style="{'color:'+color if is_top else ''}">{d}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{width_pct:.1f}%;background:{color};{'opacity:1' if is_top else 'opacity:0.45'}"></div>
                </div>
                <div class="prob-pct" style="{'color:'+color if is_top else ''}">{p*100:.1f}%</div>
            </div>
            """

        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Powered by <span>PyTorch</span> · MNIST-trained CNN · Built with Streamlit
</div>
""", unsafe_allow_html=True)
