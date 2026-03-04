# 🔢 Digit Recognizer — Streamlit

An interactive web application that lets you **draw a handwritten digit** on a canvas and instantly classifies it (0–9) using a custom **Convolutional Neural Network (CNN)** trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

Built with **PyTorch** and deployed with **Streamlit**.

---

## ✨ Features

- 🖊️ **Freehand drawing canvas** — draw any digit with your mouse or touch screen
- ⚡ **Real-time inference** — CNN prediction with confidence score shown immediately
- 📊 **Full probability breakdown** — bar chart showing softmax probabilities for all 10 digits (0–9)
- 🤖 **Self-contained model training** — if no pre-trained weights are found, the app downloads MNIST and trains a model automatically on first launch (~2 minutes)
- 🎨 **Sleek dark UI** — custom CSS with a minimal dark theme

---

## 🧠 Model Architecture

The `DigitCNN` model is a five-layer CNN with the following structure:

| Layer | Details |
|---|---|
| Conv Block 1 | Conv2d(1→32, 3×3) → ReLU → BatchNorm → Conv2d(32→32) → ReLU → MaxPool(2×2) → Dropout2d(0.25) |
| Conv Block 2 | Conv2d(32→64, 3×3) → ReLU → BatchNorm → Conv2d(64→64) → ReLU → MaxPool(2×2) → Dropout2d(0.25) |
| Conv Block 3 | Conv2d(64→128, 3×3) → ReLU → BatchNorm |
| FC Layer 1 | Linear(128×7×7 → 256) → ReLU → Dropout(0.5) |
| Output Layer | Linear(256 → 10) |

Training uses the **Adam** optimiser with a step learning-rate scheduler (×0.5 every 3 epochs) over **8 epochs** with data augmentation (random rotation, affine transforms, perspective distortion).

---

## 🗂️ Project Structure

```
Digit-recognizer-Streamlit/
├── streamlit_app.py   # Main application (model, preprocessing, UI)
├── requirements.txt   # Python dependencies
├── runtime.txt        # Python version pin (3.11)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11+** (3.11 is used in the reference deployment; see `runtime.txt`)
- [pip](https://pip.pypa.io/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Obelisk999/Digit-recognizer-Streamlit.git
cd Digit-recognizer-Streamlit

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

> **First launch:** If no pre-trained weights exist, the app will automatically download the MNIST dataset and train the model (~2 minutes). Subsequent launches load the saved weights instantly.  
> **Note (Windows):** The default model cache path is `/tmp/digit_cnn_mnist_v2.pth` (Unix/macOS). Windows users should change `MODEL_PATH` in `streamlit_app.py` to a writable directory, e.g. `C:/Users/<you>/AppData/Local/Temp/digit_cnn_mnist_v2.pth`.

---

## 🖥️ Usage

1. **Draw** any digit (0–9) in the black canvas using your mouse or stylus.
2. Click **⚡ Predict Digit** to run inference.
3. View the **predicted digit**, **confidence score**, and the **full softmax probability bar chart**.
4. Click **✕ Clear** to reset the canvas and draw again.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `torch` / `torchvision` | CNN model training and inference |
| `streamlit-drawable-canvas` | Interactive drawing canvas component |
| `Pillow` | Image preprocessing |
| `numpy` | Numerical operations |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## ⚙️ How It Works

1. The user draws a digit on a 280×280 HTML canvas (white stroke on black background).
2. The raw RGBA image is converted to greyscale and **inverted** (MNIST uses white digits on black).
3. A threshold removes faint noise; the digit bounding box is cropped with padding.
4. The crop is **resized to 20×20** (preserving aspect ratio) and **centred on a 28×28 canvas** — exactly matching MNIST input format.
5. Pixel values are normalised with MNIST mean (0.1307) and std (0.3081).
6. The tensor is passed through `DigitCNN` and a **softmax** produces per-class probabilities.
7. The argmax gives the predicted digit; the corresponding probability is the confidence score.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.
