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

# ─── MODEL DEFINITION — exact HRNet from training notebook ──────────────────────
BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        num_out = num_inchannels[i] if k == i - j - 1 else num_inchannels[j]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_out, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(num_out, momentum=BN_MOMENTUM),
                            nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem — stride=2 twice, input must be 224x224
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        stage1_out = Bottleneck.expansion * 64  # 256

        num_channels = [nc * BasicBlock.expansion for nc in [18, 36]]
        self.transition1 = self._make_transition_layer([stage1_out], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(2, BasicBlock, [4, 4], num_channels)

        num_channels = [nc * BasicBlock.expansion for nc in [18, 36, 72]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(3, BasicBlock, [4, 4, 4], num_channels, num_modules=3)

        num_channels = [nc * BasicBlock.expansion for nc in [18, 36, 72, 144]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(4, BasicBlock, [4, 4, 4, 4], num_channels, num_modules=4)

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
        self.classifier = nn.Linear(2048, 10)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        layers = []
        for i in range(len(num_channels_cur)):
            if i < len(num_channels_pre):
                if num_channels_cur[i] != num_channels_pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - len(num_channels_pre)):
                    in_ch  = num_channels_pre[-1]
                    out_ch = num_channels_cur[i] if j == i - len(num_channels_pre) else in_ch
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(layers)

    def _make_stage(self, num_branches, block, num_blocks, num_channels, num_modules=2, multi_scale_output=True):
        num_inchannels = num_channels[:]
        modules = []
        for i in range(num_modules):
            mso = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, 'SUM', mso))
            num_inchannels = modules[-1].num_inchannels
        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, pre_stage_channels):
        head_block    = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = nn.ModuleList([self._make_layer(head_block, pre_stage_channels[i], head_channels[i], 1) for i in range(len(pre_stage_channels))])
        downsamp_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(head_channels[i] * head_block.expansion, head_channels[i+1] * head_block.expansion, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(head_channels[i+1] * head_block.expansion, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
            for i in range(len(pre_stage_channels) - 1)])
        final_layer = nn.Sequential(
            nn.Conv2d(head_channels[3] * head_block.expansion, 2048, 1, 1, 0, bias=True),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        y = [t(x) if t is not None else x for t in self.transition1]
        for m in self.stage2: y = m(y)

        y2 = []
        for i, t in enumerate(self.transition2):
            src = y[-1] if i >= len(y) else y[i]
            y2.append(t(src) if t is not None else src)
        for m in self.stage3: y2 = m(y2)

        y3 = []
        for i, t in enumerate(self.transition3):
            src = y2[-1] if i >= len(y2) else y2[i]
            y3.append(t(src) if t is not None else src)
        for m in self.stage4: y3 = m(y3)

        y = self.incre_modules[0](y3[0])
        for i in range(len(self.downsamp_modules)):
            y = self.downsamp_modules[i](y) + self.incre_modules[i+1](y3[i+1])
        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
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
    try:
        obj = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)

        # Case 1: full model object saved directly
        if isinstance(obj, torch.nn.Module):
            obj.eval()
            return obj, None

        # Extract state dict
        if isinstance(obj, dict):
            sd = obj.get("model_state_dict") or obj.get("state_dict") or obj
        else:
            sd = obj

        # ── Remap checkpoint keys to match our architecture ──
        # Checkpoint stores fuse_layers downsampling as:
        #   fuse_layers.i.j.0.0.weight  (extra nn.ModuleList wrapper)
        # Our architecture stores them as:
        #   fuse_layers.i.j.0.weight    (flat nn.Sequential)
        # Similarly for transitions:
        #   transition2.2.0.0.weight → transition2.2.0.weight
        import re
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            # fuse_layers: .j.k.N.M. → .j.k.N. (collapse extra ModuleList level)
            # Pattern: fuse_layers.i.j.d.e.f → fuse_layers.i.j.concat(d*steps+e).f
            # Simpler: just strip the extra nested index for downsampling paths
            # e.g. stage2.0.fuse_layers.1.0.0.0.weight → stage2.0.fuse_layers.1.0.0.weight
            #      stage2.0.fuse_layers.1.0.0.1.weight → stage2.0.fuse_layers.1.0.1.weight
            m = re.match(r'(.*\.fuse_layers\.\d+\.\d+\.)(\d+)\.(\d+)(.*)', new_k)
            if m:
                prefix, outer, inner, suffix = m.groups()
                # outer=0 means first conv3x3 block, inner is the layer within it
                # flatten: new index = outer * 3 + inner  (each conv3x3s block has 3 layers: Conv, BN, ReLU)
                flat_idx = int(outer) * 3 + int(inner)
                new_k = f"{prefix}{flat_idx}{suffix}"

            # transitions: transition2.2.0.0.weight → transition2.2.0.weight
            #              transition3.3.0.0.weight → transition3.3.0.weight
            m2 = re.match(r'(transition\d+\.\d+\.)(\d+)\.(\d+)(.*)', new_k)
            if m2:
                prefix, outer, inner, suffix = m2.groups()
                flat_idx = int(outer) * 3 + int(inner)
                new_k = f"{prefix}{flat_idx}{suffix}"

            new_sd[new_k] = v

        model = DigitCNN()
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            # Filter out truly harmless keys (num_batches_tracked)
            real_missing = [k for k in missing if 'num_batches_tracked' not in k]
            if real_missing:
                return None, f"⚠️ Keys still missing after remapping: {real_missing[:5]}"
        model.eval()
        return model, None
    except Exception as e:
        return None, f"⚠️ Error loading model: {e}"


# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
def preprocess_image(img_array: np.ndarray) -> torch.Tensor:
    """Convert canvas RGBA → 224×224 normalised tensor (matches training pipeline)."""
    img = Image.fromarray(img_array.astype("uint8"), "RGBA").convert("L")
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


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
