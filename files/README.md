# 🤚 SignSpeak — Live ASL Detection with WLASL

A modular, production-ready American Sign Language recognition pipeline
using the **WLASL (Word-Level ASL)** dataset — real sign-language words,
not just the alphabet.

---

## 📁 Project Structure

```
asl_project/
├── module1_dataset.py          # Download, preprocess, split WLASL data
├── module2_train_cnn.py        # Conv1D feature extractor training
├── module3_lstm_nlp_frontend.py # Bi-LSTM + NLP pipeline + live HTML app
└── README.md
```

---

## 🗃 Dataset — WLASL (replaces old ASL Alphabet)

| Property       | Old dataset              | New dataset (WLASL)              |
|----------------|--------------------------|----------------------------------|
| Type           | Static images (A-Z)      | Video clips (real words)         |
| Classes        | 29 letters               | 100–2 000 glosses (words)        |
| Kaggle slug    | `grassknoted/asl-alphabet`| `risangbaskoro/wlasl-processed` |
| Keypoints      | CNN on 64×64 images      | MediaPipe Holistic (pose+hands)  |
| Temporal info  | None                     | 30-frame sequences               |

---

## 🚀 Run Order (Google Colab, GPU runtime)

### Step 1 — Set your Kaggle credentials in `module1_dataset.py`
```python
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"]      = "your_api_key"
```
Then run **Module 1** — it downloads WLASL, extracts MediaPipe keypoints,
and saves `splits.npz` + `class_map.json`.

### Step 2 — Run Module 2
Trains a **Conv1D** feature extractor on the keypoint sequences.
Outputs `best_cnn.keras` and `cnn_features.npz` (embeddings for LSTM).

### Step 3 — Run Module 3
- Trains a **Bidirectional LSTM** on the CNN embeddings
- Applies a rule-based **NLP pipeline** (gloss → English sentence)
- Serves a beautiful **live HTML frontend** via Flask + localtunnel

---

## 🧠 Architecture

```
Video frame
    │
    ▼
MediaPipe Holistic ──→ Keypoints (1662-D per frame)
    │
    ▼  (30 frames stacked → sequence)
Conv1D × 3 blocks ──→ Embedding (256-D)
    │
    ▼
Bi-LSTM × 2 ──→ Classification (top-N words)
    │
    ▼
NLP pipeline ──→ English sentence
    │
    ▼
Live HTML UI (camera + sign display + text output)
```

---

## 📦 Dependencies

```
tensorflow >= 2.13
mediapipe  >= 0.10
opencv-python-headless
numpy  scikit-learn  matplotlib
flask  localtunnel (npm)
kaggle
```

Install in Colab:
```bash
!pip install tensorflow mediapipe opencv-python-headless scikit-learn flask kaggle -q
!npm install -g localtunnel -q
```

---

## 🖥 Frontend Features

- **MediaPipe Holistic** — pose + both hands detected simultaneously
- **Hold-to-commit** — hold a sign steady for ~18 frames to add it to text
- **NLP sentence** — converts raw gloss sequence to readable English
- **Copy / Delete / Space / Clear** controls
- **Last 12 signs** history panel
- Mirrored camera feed with skeleton overlay

---

## 📌 Notes

- For full 2000-word WLASL, set `TOP_N = 2000` in Module 1 (needs ~4 h on GPU).
- The frontend classifier is rule-based for demo purposes;
  wire it to a `/predict` Flask endpoint for real model inference.
- localtunnel may ask for your public IP as a password — run `curl ifconfig.me`.
