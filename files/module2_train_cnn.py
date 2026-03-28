"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 2 — CNN Feature Extractor Training                   ║
║  Input  : splits.npz  (from Module 1)                        ║
║  Output : best_cnn.keras  +  cnn_features.npz               ║
╚══════════════════════════════════════════════════════════════╝

NOTE: WLASL keypoints are 1-D sequences (time × features).
      We treat each frame's keypoint vector as a 1-D signal
      and use Conv1D layers as the feature extractor.
      The extracted embeddings are fed to the LSTM in Module 3.
"""

# ─── CELL 1 : Imports & GPU check ────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import json, os

print("TensorFlow :", tf.__version__)
print("GPUs       :", tf.config.list_physical_devices("GPU"))
tf.keras.backend.clear_session()

# ─── CELL 2 : Load splits ────────────────────────────────────────────────────
SPLITS_PATH  = "/content/wlasl/splits.npz"
CLASS_MAP    = "/content/wlasl/class_map.json"

data     = np.load(SPLITS_PATH)
X_train  = data["X_train"]   # (N, SEQ_LEN, 1662)
X_val    = data["X_val"]
X_test   = data["X_test"]
y_train  = data["y_train"].astype("int32")
y_val    = data["y_val"].astype("int32")
y_test   = data["y_test"].astype("int32")

with open(CLASS_MAP) as f:
    cmap        = json.load(f)
    IDX2CLASS   = {int(k): v for k, v in cmap["idx2class"].items()}

NUM_CLASSES  = len(IDX2CLASS)
SEQ_LEN      = X_train.shape[1]   # 30
FEATURE_DIM  = X_train.shape[2]   # 1662

print(f"Train  : {X_train.shape}")
print(f"Val    : {X_val.shape}")
print(f"Test   : {X_test.shape}")
print(f"Classes: {NUM_CLASSES}")

# ─── CELL 3 : Normalise ──────────────────────────────────────────────────────
"""
Per-feature standardisation across the training set.
Saved so we can apply the same transform at inference time.
"""
mean = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, 1662)
std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

np.save("/content/wlasl/norm_mean.npy", mean)
np.save("/content/wlasl/norm_std.npy",  std)
print("✅ Normalisation stats saved.")

# ─── CELL 4 : One-hot labels for the CNN head ────────────────────────────────
y_train_oh = to_categorical(y_train, NUM_CLASSES)
y_val_oh   = to_categorical(y_val,   NUM_CLASSES)

# ─── CELL 5 : Build Conv1D feature-extractor + classifier ────────────────────
"""
Architecture:
  Conv1D × 3  (with BatchNorm + MaxPool) → GlobalAvgPool → Dense head

The output of GlobalAveragePooling1D is the embedding we pass to LSTM.
"""

def build_cnn(seq_len, feat_dim, num_classes):
    inp = Input(shape=(seq_len, feat_dim), name="keypoint_input")

    # ── Block 1 ──────────────────────────────────────────────
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # ── Block 2 ──────────────────────────────────────────────
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # ── Block 3 ──────────────────────────────────────────────
    x = Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    embedding = GlobalAveragePooling1D(name="embedding")(x)   # (256,) per sample

    # ── Classifier head ──────────────────────────────────────
    x = Dense(256, activation="relu")(embedding)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="ASL_Conv1D")
    return model


cnn_model = build_cnn(SEQ_LEN, FEATURE_DIM, NUM_CLASSES)
cnn_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)
cnn_model.summary()

# ─── CELL 6 : Train CNN ───────────────────────────────────────────────────────
EPOCHS     = 40
BATCH_SIZE = 32

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("/content/wlasl/best_cnn.keras",
                    save_best_only=True, monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1)
]

history = cnn_model.fit(
    X_train, y_train_oh,
    validation_data = (X_val, y_val_oh),
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    callbacks       = callbacks,
    verbose         = 1
)

# ─── CELL 7 : Evaluate on test set ───────────────────────────────────────────
y_test_oh = to_categorical(y_test, NUM_CLASSES)
loss, acc  = cnn_model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n🎯 Test Accuracy : {acc * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")

# ─── CELL 8 : Training curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend()

plt.tight_layout()
plt.savefig("/content/wlasl/training_curves.png", dpi=120)
plt.show()

# ─── CELL 9 : Extract embeddings for Module 3 ────────────────────────────────
"""
Build a sub-model that stops at the 'embedding' layer.
These 256-D vectors are the input sequences for the LSTM.
"""
extractor = Model(
    inputs  = cnn_model.input,
    outputs = cnn_model.get_layer("embedding").output,
    name    = "feature_extractor"
)
extractor.trainable = False

def extract(X, batch_size=128):
    out = []
    for i in range(0, len(X), batch_size):
        out.append(extractor(X[i: i + batch_size], training=False).numpy())
    return np.concatenate(out, axis=0)

print("Extracting train embeddings …")
E_train = extract(X_train)
print("Extracting val embeddings …")
E_val   = extract(X_val)
print("Extracting test embeddings …")
E_test  = extract(X_test)

print(f"Embedding shapes — Train {E_train.shape} | Val {E_val.shape} | Test {E_test.shape}")

np.savez_compressed("/content/wlasl/cnn_features.npz",
    E_train=E_train, y_train=y_train,
    E_val=E_val,     y_val=y_val,
    E_test=E_test,   y_test=y_test)

cnn_model.save("/content/wlasl/best_cnn.keras")
extractor.save("/content/wlasl/feature_extractor.keras")
print("✅ Module 2 complete!  →  run Module 3 next.")
