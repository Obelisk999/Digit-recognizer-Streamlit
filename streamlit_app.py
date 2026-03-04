import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered", initial_sidebar_state="collapsed")

# ─── MODEL ──────────────────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

MODEL_PATH = "/tmp/digit_cnn_mnist_v2.pth"

@st.cache_resource(show_spinner=False)
def load_model():
    model = DigitCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, None
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        tf = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        tf_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST("/tmp/mnist", train=True,  download=True, transform=tf)
        val_ds   = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=tf_val)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        for epoch in range(8):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        torch.save(model.state_dict(), MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"⚠️ Failed to train model: {e}"

def preprocess_image(img_array):
    img = Image.fromarray(img_array.astype("uint8"), "RGBA").convert("L")
    img = ImageOps.invert(img)
    # Threshold to remove noise — anything faint becomes black background
    arr = np.array(img)
    arr = (arr > 50).astype(np.uint8) * 255
    img = Image.fromarray(arr)
    bbox = img.getbbox()
    if bbox is None:
        # Empty canvas — return blank
        arr = np.zeros((28, 28), dtype=np.float32)
        arr = (arr - 0.1307) / 0.3081
        return torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    # Add padding around the digit before resizing
    pad = 20
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(arr.shape[1], x1 + pad), min(arr.shape[0], y1 + pad)
    img = img.crop((x0, y0, x1, y1))
    # Resize keeping aspect ratio, fit inside 20x20
    img.thumbnail((20, 20), Image.LANCZOS)
    # Centre on 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    offset_x = (28 - img.width)  // 2
    offset_y = (28 - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred]), probs

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
:root{--bg:#0d0d0f;--surface:#16161a;--border:#2a2a32;--accent:#e8ff47;--accent2:#ff6b6b;--text:#f0f0f0;--muted:#6b6b7a}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text)}
.main .block-container{max-width:720px;padding:2rem 2rem 4rem}
.header-wrap{text-align:center;padding:2.5rem 0 1.5rem}
.header-badge{display:inline-block;background:var(--accent);color:#000;font-family:'Space Mono',monospace;font-size:.65rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;padding:4px 12px;border-radius:2px;margin-bottom:1rem}
.header-title{font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;letter-spacing:-.03em;line-height:1;margin:0}
.header-title span{color:var(--accent)}
.header-sub{font-size:.95rem;color:var(--muted);margin-top:.6rem;font-weight:300}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1.2rem}
.card-title{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem}
.pred-box{display:flex;align-items:center;gap:1.5rem;background:#0d0d0f;border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.5rem}
.pred-digit{font-family:'Space Mono',monospace;font-size:5rem;font-weight:700;color:var(--accent);line-height:1;min-width:70px;text-align:center}
.pred-meta{flex:1}
.pred-label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;font-family:'Space Mono',monospace}
.pred-conf{font-size:2rem;font-weight:600;color:var(--text);line-height:1.2}
.conf-bar-wrap{width:100%;height:4px;background:var(--border);border-radius:2px;margin-top:8px;overflow:hidden}
.conf-bar{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),#b8d900)}
.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.prob-digit-label{font-family:'Space Mono',monospace;font-size:.75rem;color:var(--muted);width:16px;text-align:right}
.prob-bar-bg{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:4px}
.prob-pct{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--muted);width:42px;text-align:right}
.instructions{display:flex;gap:1rem;flex-wrap:wrap}
.step{display:flex;align-items:flex-start;gap:8px;flex:1;min-width:140px}
.step-num{font-family:'Space Mono',monospace;font-size:.65rem;background:var(--accent);color:#000;width:18px;height:18px;border-radius:3px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}
.step-text{font-size:.82rem;color:var(--muted);line-height:1.4}
div.stButton>button{width:100%;background:var(--accent)!important;color:#000!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;font-weight:700!important;letter-spacing:.1em!important;text-transform:uppercase!important;border:none!important;border-radius:6px!important;padding:.65rem!important}
div.stButton>button:hover{opacity:.85!important}
.error-box{background:#1a0f0f;border:1px solid var(--accent2);border-radius:8px;padding:1rem 1.2rem;color:var(--accent2);font-size:.9rem}
.footer{text-align:center;font-size:.72rem;color:var(--muted);margin-top:3rem;font-family:'Space Mono',monospace}
.footer span{color:var(--accent)}
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
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
with st.spinner("⚙️ Loading model… (first launch trains on MNIST, ~2 min)"):
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
        <div class="step"><div class="step-num">3</div><div class="step-text">See the prediction and confidence score</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text">Hit <strong>Clear</strong> to draw again</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── CANVAS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Draw here</div>', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)", stroke_width=18, stroke_color="#FFFFFF",
    background_color="#000000", height=280, width=280,
    drawing_mode="freedraw", key="canvas",
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
            time.sleep(0.2)

        COLORS = ["#e8ff47","#a8ff78","#78ffd6","#48c6ef","#6f86d6","#f093fb","#f5576c","#fd746c","#ff9a44","#ffd452"]

        st.markdown(f"""
        <div class="card" style="margin-top:1.2rem">
            <div class="card-title">Prediction</div>
            <div class="pred-box">
                <div class="pred-digit">{pred}</div>
                <div class="pred-meta">
                    <div class="pred-label">Confidence</div>
                    <div class="pred-conf">{conf*100:.1f}%</div>
                    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf*100:.1f}%"></div></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">All class probabilities</div>', unsafe_allow_html=True)
        bars_html = ""
        for d in range(10):
            p = probs[d]
            is_top = d == pred
            color = COLORS[d]
            bars_html += f"""
            <div class="prob-row" style="{'background:#ffffff08;border-radius:6px;padding:2px 4px;' if is_top else ''}">
                <div class="prob-digit-label" style="{'color:'+color if is_top else ''}">{d}</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{color};{'opacity:1' if is_top else 'opacity:0.45'}"></div></div>
                <div class="prob-pct" style="{'color:'+color if is_top else ''}">{p*100:.1f}%</div>
            </div>"""
        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Powered by <span>PyTorch</span> · MNIST-trained CNN · Built with Streamlit</div>', unsafe_allow_html=True)
