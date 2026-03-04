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

# ─── MODEL DEFINITION — HRNet for digit classification ──────────────────────────
# Reconstructed from the checkpoint keys in best_model.pth

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x) if self.downsample else x
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample: residual = self.downsample(x)
        return F.relu(out + residual)


class HRModule(nn.Module):
    def __init__(self, branches, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(channels[i], channels[i]) for _ in range(4)])
            for i in range(branches)
        ])
        self.fuse_layers = nn.ModuleList()
        for i in range(branches):
            fuse = nn.ModuleList()
            for j in range(branches):
                if j > i:
                    fuse.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i])
                    ))
                elif j < i:
                    ops = []
                    for k in range(i - j):
                        ops += [nn.Conv2d(channels[j], channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(channels[j])]
                    fuse.append(nn.Sequential(*ops))
                else:
                    fuse.append(None)
            self.fuse_layers.append(fuse)

    def forward(self, x):
        branches_out = [b(x[i]) for i, b in enumerate(self.branches)]
        out = []
        for i, fuse in enumerate(self.fuse_layers):
            y = branches_out[i]
            for j, f in enumerate(fuse):
                if f is None: continue
                bj = branches_out[j]
                if j > i:
                    bj = F.interpolate(f(bj), size=y.shape[-2:], mode='nearest')
                    y = y + bj
                else:
                    y = y + F.interpolate(f(bj), size=y.shape[-2:], mode='nearest') if bj.shape[-1] != y.shape[-1] else y + f(bj)
            out.append(F.relu(y))
        return out


class DigitCNN(nn.Module):
    """HRNet-W18-Small adapted for 28x28 grayscale digit classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # Layer1 — 4 Bottleneck blocks, in=64 → out=256
        def make_layer1():
            layers = []
            ds = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
            layers.append(Bottleneck(64, 64, downsample=ds))
            for _ in range(3):
                layers.append(Bottleneck(256, 64))
            return nn.Sequential(*layers)
        self.layer1 = make_layer1()

        # Transition 1: 256 → [18, 36]
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 18, 3, padding=1, bias=False), nn.BatchNorm2d(18)),
            nn.Sequential(nn.Sequential(nn.Conv2d(256, 36, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(36)))
        ])

        # Stage 2: 2 branches [18, 36], 2 modules
        self.stage2 = nn.Sequential(
            HRModule(2, [18, 36]),
            HRModule(2, [18, 36]),
        )

        # Transition 2: add branch [72]
        self.transition2 = nn.ModuleList([
            None, None,
            nn.Sequential(nn.Conv2d(36, 72, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(72))
        ])

        # Stage 3: 3 branches [18, 36, 72], 3 modules
        self.stage3 = nn.Sequential(
            HRModule(3, [18, 36, 72]),
            HRModule(3, [18, 36, 72]),
            HRModule(3, [18, 36, 72]),
        )

        # Transition 3: add branch [144]
        self.transition3 = nn.ModuleList([
            None, None, None,
            nn.Sequential(nn.Conv2d(72, 144, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(144))
        ])

        # Stage 4: 4 branches [18, 36, 72, 144], 4 modules
        self.stage4 = nn.Sequential(
            HRModule(4, [18, 36, 72, 144]),
            HRModule(4, [18, 36, 72, 144]),
            HRModule(4, [18, 36, 72, 144]),
            HRModule(4, [18, 36, 72, 144]),
        )

        # Classification head
        pre_stage_channels = [18, 36, 72, 144]
        head_channels = [32, 64, 128, 256]

        self.incre_modules = nn.ModuleList()
        for i, (inc, hc) in enumerate(zip(pre_stage_channels, head_channels)):
            ds = nn.Sequential(nn.Conv2d(inc, hc * 4, 1, bias=False), nn.BatchNorm2d(hc * 4))
            self.incre_modules.append(nn.Sequential(Bottleneck(inc, hc, downsample=ds)))

        self.downsamp_modules = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512)),
            nn.Sequential(nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(1024)),
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, bias=True),
            nn.BatchNorm2d(2048),
        )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        y = [t(x) if t else x for t in self.transition1]
        for m in self.stage2: y = m(y)

        y2 = [y[0], y[1], self.transition2[2](y[1])]
        for m in self.stage3: y2 = m(y2)

        y3 = [y2[0], y2[1], y2[2], self.transition3[3](y2[2])]
        for m in self.stage4: y3 = m(y3)

        # Head
        y = self.incre_modules[0](y3[0])
        for i in range(3):
            y = self.downsamp_modules[i](y) + self.incre_modules[i+1](y3[i+1])

        y = F.relu(self.final_layer(y))
        y = F.adaptive_avg_pool2d(y, 1).view(y.size(0), -1)
        return self.classifier(y)


# ─── DOWNLOAD MODEL FROM KAGGLE ─────────────────────────────────────────────────
MODEL_PATH = "/tmp/best_model.pth"

def download_model_from_kaggle():
    """Download best_model.pth using the Kaggle CLI (stable across API versions)."""
    if os.path.exists(MODEL_PATH):
        return None

    import json, subprocess

    # ── Write credentials from Streamlit secrets ──
    try:
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
    except KeyError as e:
        return f"⚠️ Missing Streamlit secret: {e}. Add KAGGLE_USERNAME and KAGGLE_KEY in app Secrets."

    # ── Download via CLI ──
    # kaggle models instances versions download <owner>/<model>/<framework>/<instance>/<version>
    cmd = [
        "kaggle", "models", "instances", "versions", "download",
        "taitruong256/hrnet-model/pytorch/default/1",
        "--path", "/tmp",
        "--untar",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"⚠️ Kaggle CLI error: {result.stderr.strip()}"

    # ── Find the file ──
    for root, _, files in os.walk("/tmp"):
        for fname in files:
            if fname == "best_model.pth":
                src = os.path.join(root, fname)
                if src != MODEL_PATH:
                    os.rename(src, MODEL_PATH)
                return None

    return "⚠️ `best_model.pth` not found after extracting. Check the model path on Kaggle."


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    err = download_model_from_kaggle()
    if err:
        return None, err

    model = DigitCNN()
    try:
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state

        # Use strict=False: loads all matching keys, skips mismatched fuse layer nesting
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Only fail if core layers are missing (not fuse_layers)
        critical_missing = [k for k in missing if "fuse_layer" not in k]
        if critical_missing:
            return None, f"⚠️ Critical keys missing from model: {critical_missing[:5]}"
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
