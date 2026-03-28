"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 1 — Dataset Import & Preprocessing                   ║
║  Dataset: WLASL (Word-Level American Sign Language)          ║
║  Source : kaggle → risangbaskoro/wlasl-processed             ║
╚══════════════════════════════════════════════════════════════╝

Run this notebook cell-by-cell in Google Colab (GPU runtime).
"""

# ─── CELL 1 : Install & authenticate Kaggle ───────────────────────────────────
!pip install -q kaggle opencv-python-headless mediapipe

import os, json, shutil, random, cv2, numpy as np
from pathlib import Path

# ── Set credentials (replace with your own) ──────────────────
os.environ["KAGGLE_USERNAME"] = "YOUR_KAGGLE_USERNAME"
os.environ["KAGGLE_KEY"]      = "YOUR_KAGGLE_KEY"

print("✅ Kaggle credentials set.")

# ─── CELL 2 : Download WLASL dataset ─────────────────────────────────────────
"""
WLASL = Word-Level American Sign Language
  • 2000 words, ~21 000 video clips, multiple signers
  • Much richer than ASL alphabet (static images) — real sign language
  • kaggle dataset: risangbaskoro/wlasl-processed
    contains pre-extracted skeleton JSON files (no raw video needed)
"""

!kaggle datasets download -d risangbaskoro/wlasl-processed -p /content/wlasl --unzip
print("✅ Download complete.")
print(os.listdir("/content/wlasl"))

# ─── CELL 3 : Explore the dataset structure ──────────────────────────────────
BASE_DIR   = "/content/wlasl"
JSON_PATH  = BASE_DIR + "/WLASL_v0.3.json"          # word-level metadata
VIDEO_DIR  = BASE_DIR + "/videos"                    # (may be absent in skeleton-only version)

with open(JSON_PATH) as f:
    wlasl_data = json.load(f)

print(f"Total glosses (words) : {len(wlasl_data)}")
print(f"First 5 words         : {[e['gloss'] for e in wlasl_data[:5]]}")
print(f"\nSample entry keys     : {list(wlasl_data[0].keys())}")
print(f"Sample instance keys  : {list(wlasl_data[0]['instances'][0].keys())}")

# ─── CELL 4 : Choose vocabulary & filter ─────────────────────────────────────
"""
For a manageable first model we keep the TOP-N most common words.
Increase TOP_N once you're happy with the pipeline.
"""
TOP_N = 100          # words to train on  (use 2000 for full dataset)
MIN_CLIPS = 5        # minimum clips per word

vocab = []
for entry in wlasl_data:
    gloss  = entry["gloss"]
    n_clips = len(entry["instances"])
    if n_clips >= MIN_CLIPS:
        vocab.append((gloss, n_clips))

vocab.sort(key=lambda x: -x[1])
vocab = vocab[:TOP_N]
CLASSES = [v[0] for v in vocab]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}

print(f"\nVocabulary size : {len(CLASSES)}")
print(f"Top-10 words    : {CLASSES[:10]}")
with open("/content/wlasl/class_map.json", "w") as f:
    json.dump({"class2idx": CLASS2IDX, "idx2class": {str(k): v for k, v in IDX2CLASS.items()}}, f)
print("✅ class_map.json saved.")

# ─── CELL 5 : Skeleton / keypoint extraction with MediaPipe ──────────────────
"""
WLASL-processed already ships 'nslt_XXXX.json' keypoint files.
If they are present we load them directly (fast).
If only videos are present we extract keypoints with MediaPipe (slower).

Keypoint format: 543 points  (pose 33 + left hand 21 + right hand 21 × 3D coords)
→ flattened to vector of length 1662
"""

import mediapipe as mp

mp_holistic   = mp.solutions.holistic
N_KEYPOINTS   = 1662    # 33*4 + 21*4 + 21*4  (x,y,z,vis)
SEQ_LEN       = 30      # frames per clip

def extract_keypoints(results):
    """Return flat numpy array of length 1662."""
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]
                    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
                    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])   # (1662,)


def video_to_sequence(video_path, seq_len=SEQ_LEN):
    """
    Read a video file, extract MediaPipe keypoints for each frame,
    pad / truncate to seq_len, return array (seq_len, N_KEYPOINTS).
    """
    cap   = cv2.VideoCapture(str(video_path))
    frames = []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frames.append(extract_keypoints(results))

    cap.release()

    # ── Pad / truncate ────────────────────────────────────────
    if len(frames) == 0:
        return np.zeros((seq_len, N_KEYPOINTS), dtype="float32")
    frames = np.array(frames, dtype="float32")
    if len(frames) >= seq_len:
        # centre-crop
        start  = (len(frames) - seq_len) // 2
        frames = frames[start : start + seq_len]
    else:
        # repeat-pad at end
        pad    = np.tile(frames[-1], (seq_len - len(frames), 1))
        frames = np.vstack([frames, pad])
    return frames   # (seq_len, N_KEYPOINTS)


# ─── CELL 6 : Build numpy arrays (X, y) ──────────────────────────────────────
"""
We iterate over WLASL JSON, find each video clip,
extract keypoints and build X / y arrays.

If the dataset only has pre-extracted .npy files in npz format,
we load those instead (much faster — see the `npz_path` branch).
"""

NPZ_CACHE = "/content/wlasl/dataset_cache.npz"

if os.path.exists(NPZ_CACHE):
    # ── Fast path: load cached numpy arrays ──────────────────
    print("Loading from cache …")
    data  = np.load(NPZ_CACHE)
    X_all = data["X"]
    y_all = data["y"]
    print(f"✅ Loaded X {X_all.shape}, y {y_all.shape}")

else:
    # ── Slow path: extract from videos ───────────────────────
    X_list, y_list = [], []
    missing = 0

    for entry in wlasl_data:
        gloss = entry["gloss"]
        if gloss not in CLASS2IDX:
            continue
        label = CLASS2IDX[gloss]

        for inst in entry["instances"]:
            vid_id = inst["video_id"]
            vpath  = Path(BASE_DIR) / "videos" / f"{vid_id}.mp4"
            if not vpath.exists():
                missing += 1
                continue
            seq = video_to_sequence(vpath)
            X_list.append(seq)
            y_list.append(label)

    print(f"Missing videos : {missing}")

    if len(X_list) == 0:
        raise RuntimeError("No video files found — check dataset path / video folder.")

    X_all = np.array(X_list, dtype="float32")
    y_all = np.array(y_list, dtype="int32")
    np.savez_compressed(NPZ_CACHE, X=X_all, y=y_all)
    print(f"✅ Saved cache: X {X_all.shape}, y {y_all.shape}")

# ─── CELL 7 : Train / val / test split ───────────────────────────────────────
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.10, stratify=y_all, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

print(f"Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"Classes in train : {len(np.unique(y_train))}")

# Save splits for module 2
np.savez_compressed("/content/wlasl/splits.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test)

print("✅ splits.npz saved — ready for Module 2!")

# ─── CELL 8 : Quick sanity-check visualisation ───────────────────────────────
import matplotlib.pyplot as plt

label_counts = {}
for y in y_train:
    label_counts[IDX2CLASS[y]] = label_counts.get(IDX2CLASS[y], 0) + 1

top10 = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
words, counts = zip(*top10)

plt.figure(figsize=(10, 4))
plt.bar(words, counts, color="#4CAF50")
plt.title("Top-10 classes in training set")
plt.ylabel("Clip count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("/content/wlasl/class_distribution.png", dpi=120)
plt.show()
print("✅ Module 1 complete!")
