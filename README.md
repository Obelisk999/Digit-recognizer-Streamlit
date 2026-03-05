<div align="center">

# 🔢 Digit.AI — Real-Time Handwritten Digit Recognizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MNIST-Dataset-00B4D8?style=for-the-badge&logo=databricks&logoColor=white" alt="MNIST"/>
  <img src="https://img.shields.io/badge/License-Open%20Source-brightgreen?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <b>Draw any digit (0–9) on a canvas and watch a CNN classify it in real time ⚡</b><br/>
  <i>Powered by a custom PyTorch CNN · Trained on MNIST · Deployed with Streamlit</i>
</p>

</div>

---

## ✨ Features

| Feature | Description |
|:-------:|-------------|
| 🖊️ **Free-Draw Canvas** | Draw any digit with your mouse or touchscreen on a 280×280 canvas |
| ⚡ **Real-Time Inference** | Instant CNN prediction with confidence score |
| 📊 **Probability Distribution** | Softmax bar chart for all 10 digit classes (0–9) |
| 🤖 **Self-Training Model** | Auto-downloads MNIST and trains from scratch on first launch (~2 min) |
| 🎨 **Sleek Dark UI** | Custom CSS with a minimalist dark theme using Space Mono & DM Sans |

---

## 🧠 Model Architecture

> **DigitCNN** — a 5-block Convolutional Neural Network trained end-to-end on MNIST

```
Input (1×28×28)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Conv Block 1 │ Conv2d(1→32, 3×3) → ReLU → BN              │
│               │ Conv2d(32→32, 3×3) → ReLU → MaxPool → Drop  │
├─────────────────────────────────────────────────────────────┤
│  Conv Block 2 │ Conv2d(32→64, 3×3) → ReLU → BN              │
│               │ Conv2d(64→64, 3×3) → ReLU → MaxPool → Drop  │
├─────────────────────────────────────────────────────────────┤
│  Conv Block 3 │ Conv2d(64→128, 3×3) → ReLU → BatchNorm      │
├─────────────────────────────────────────────────────────────┤
│  FC Layer     │ Linear(128×7×7 → 256) → ReLU → Dropout(0.5) │
├─────────────────────────────────────────────────────────────┤
│  Output       │ Linear(256 → 10)                             │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Softmax → Predicted Digit + Confidence Score
```

**Training details:**
- 🔧 Optimizer: **Adam** (lr = 1e-3)
- 📉 Scheduler: **StepLR** (×0.5 every 3 epochs)
- 🔁 Epochs: **8**
- 🎲 Augmentation: random rotation, affine transforms, perspective distortion

---

## 🗂️ Project Structure

```
Digit-recognizer-Streamlit/
├── 📄 streamlit_app.py   ← Main app (model, preprocessing, UI)
├── 📦 requirements.txt   ← Python dependencies
├── 🐍 runtime.txt        ← Python version pinned to 3.11
└── 📖 README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+** — see `runtime.txt`
- [pip](https://pip.pypa.io/)

### Install & Run

```bash
# 1. Clone the repo
git clone https://github.com/Obelisk999/Digit-recognizer-Streamlit.git
cd Digit-recognizer-Streamlit

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app 🚀
streamlit run streamlit_app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501) and start drawing!

> **💡 First launch:** The app will automatically download MNIST (~11 MB) and train the model (~2 min). Subsequent launches load the saved weights instantly.
>
> **🪟 Windows note:** The default model cache path is `/tmp/digit_cnn_mnist_v2.pth`. Windows users should update `MODEL_PATH` in `streamlit_app.py` to a writable path, e.g. `C:/Users/<you>/AppData/Local/Temp/digit_cnn_mnist_v2.pth`.

---

## 🖥️ How to Use

```
1. ✏️  Draw any digit (0–9) on the black canvas
2. ⚡  Click "Predict Digit" to run the CNN
3. 🎯  See the predicted digit + confidence score
4. 📊  Explore the full probability distribution
5. 🔄  Hit "Clear" to draw again
```

---

## ⚙️ Under the Hood

```
User Drawing (280×280 RGBA canvas)
         │
         ▼
  Convert to Grayscale & Invert
  (MNIST format: white digit on black background)
         │
         ▼
  Threshold (>50) → Remove noise
         │
         ▼
  Crop bounding box + 20px padding
         │
         ▼
  Resize to 20×20 (aspect-ratio preserved)
         │
         ▼
  Center-paste onto 28×28 black canvas
         │
         ▼
  Normalize (mean=0.1307, std=0.3081)
         │
         ▼
  DigitCNN → Softmax → argmax = Predicted Digit 🎯
```

---

## 📦 Dependencies

| Package | Role |
|---------|------|
| `streamlit` | Web app framework |
| `torch` / `torchvision` | CNN training & inference |
| `streamlit-drawable-canvas` | Interactive drawing component |
| `Pillow` | Image preprocessing |
| `numpy` | Numerical operations |

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions, ideas, and bug reports are welcome! Feel free to:

1. 🍴 Fork the project
2. 🔧 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

---

## 📄 License

This project is **open source** — feel free to use, modify, and distribute it.

---

<div align="center">

Made with ❤️ and ☕ · Powered by <b>PyTorch</b> · Built with <b>Streamlit</b>

⭐ If you find this project useful, give it a star!

</div>
