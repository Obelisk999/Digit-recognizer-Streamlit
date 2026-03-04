# 🔢 Reconnaissance de Chiffres — Streamlit

Une application web interactive qui vous permet de **dessiner un chiffre à la main** sur un canvas et de le classifier instantanément (0–9) grâce à un **Réseau de Neurones Convolutif (CNN)** personnalisé entraîné sur le jeu de données [MNIST](http://yann.lecun.com/exdb/mnist/).

Développée avec **PyTorch** et déployée avec **Streamlit**.

---

## ✨ Fonctionnalités

- 🖊️ **Canvas de dessin libre** — dessinez n'importe quel chiffre avec la souris ou un écran tactile
- ⚡ **Inférence en temps réel** — prédiction du CNN avec score de confiance affiché immédiatement
- 📊 **Distribution complète des probabilités** — graphique en barres montrant les probabilités softmax pour les 10 chiffres (0–9)
- 🤖 **Entraînement autonome du modèle** — si aucun poids pré-entraîné n'est trouvé, l'application télécharge MNIST et entraîne un modèle automatiquement au premier lancement (~2 minutes)
- 🎨 **Interface sombre élégante** — CSS personnalisé avec un thème sombre minimaliste

---

## 🧠 Architecture du Modèle

Le modèle `DigitCNN` est un CNN à cinq couches avec la structure suivante :

| Couche | Détails |
|---|---|
| Bloc Conv 1 | Conv2d(1→32, 3×3) → ReLU → BatchNorm → Conv2d(32→32) → ReLU → MaxPool(2×2) → Dropout2d(0.25) |
| Bloc Conv 2 | Conv2d(32→64, 3×3) → ReLU → BatchNorm → Conv2d(64→64) → ReLU → MaxPool(2×2) → Dropout2d(0.25) |
| Bloc Conv 3 | Conv2d(64→128, 3×3) → ReLU → BatchNorm |
| Couche FC 1 | Linear(128×7×7 → 256) → ReLU → Dropout(0.5) |
| Couche de Sortie | Linear(256 → 10) |

L'entraînement utilise l'optimiseur **Adam** avec un planificateur de taux d'apprentissage par paliers (×0,5 toutes les 3 époques) sur **8 époques** avec augmentation de données (rotation aléatoire, transformations affines, distorsion de perspective).

---

## 🗂️ Structure du Projet

```
Digit-recognizer-Streamlit/
├── streamlit_app.py   # Application principale (modèle, prétraitement, interface)
├── requirements.txt   # Dépendances Python
├── runtime.txt        # Version Python fixée (3.11)
└── README.md
```

---

## 🚀 Démarrage Rapide

### Prérequis

- Python **3.11+** (3.11 est utilisé dans le déploiement de référence ; voir `runtime.txt`)
- [pip](https://pip.pypa.io/)

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Obelisk999/Digit-recognizer-Streamlit.git
cd Digit-recognizer-Streamlit

# 2. (Optionnel) Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'Application

```bash
streamlit run streamlit_app.py
```

Ouvrez votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

> **Premier lancement :** Si aucun poids pré-entraîné n'existe, l'application téléchargera automatiquement le jeu de données MNIST et entraînera le modèle (~2 minutes). Les lancements suivants chargent les poids sauvegardés instantanément.  
> **Remarque (Windows) :** Le chemin de cache du modèle par défaut est `/tmp/digit_cnn_mnist_v2.pth` (Unix/macOS). Les utilisateurs Windows doivent modifier `MODEL_PATH` dans `streamlit_app.py` pour pointer vers un répertoire accessible en écriture, par exemple `C:/Users/<vous>/AppData/Local/Temp/digit_cnn_mnist_v2.pth`.

---

## 🖥️ Utilisation

1. **Dessinez** n'importe quel chiffre (0–9) dans le canvas noir avec la souris ou un stylet.
2. Cliquez sur **⚡ Predict Digit** pour lancer l'inférence.
3. Consultez le **chiffre prédit**, le **score de confiance** et le **graphique complet des probabilités softmax**.
4. Cliquez sur **✕ Clear** pour réinitialiser le canvas et dessiner à nouveau.

---

## 📦 Dépendances

| Package | Rôle |
|---|---|
| `streamlit` | Framework d'application web |
| `torch` / `torchvision` | Entraînement et inférence du CNN |
| `streamlit-drawable-canvas` | Composant canvas de dessin interactif |
| `Pillow` | Prétraitement des images |
| `numpy` | Opérations numériques |

Installez toutes les dépendances avec :

```bash
pip install -r requirements.txt
```

---

## ⚙️ Fonctionnement

1. L'utilisateur dessine un chiffre sur un canvas HTML 280×280 (trait blanc sur fond noir).
2. L'image RGBA brute est convertie en niveaux de gris et **inversée** (MNIST utilise des chiffres blancs sur fond noir).
3. Un seuillage supprime le bruit léger ; la boîte englobante du chiffre est recadrée avec un rembourrage.
4. Le recadrage est **redimensionné à 20×20** (en conservant les proportions) et **centré sur un canvas 28×28** — correspondant exactement au format d'entrée MNIST.
5. Les valeurs de pixels sont normalisées avec la moyenne MNIST (0,1307) et l'écart-type (0,3081).
6. Le tenseur est passé dans `DigitCNN` et un **softmax** produit les probabilités par classe.
7. L'argmax donne le chiffre prédit ; la probabilité correspondante est le score de confiance.

---

## 📄 Licence

Ce projet est open source. N'hésitez pas à l'utiliser, le modifier et le distribuer.
