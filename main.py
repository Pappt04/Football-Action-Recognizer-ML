"""
Automatsko prepoznavanje sportskih akcija iz slika
Tim: Stefan Paunović (RA 27/2022) i Tamás Papp (RA 4/2022)
Asistent: Teodor Vidaković

Pokretanje:
    source venv/bin/activate
    python main.py           # puno treniranje (~5-7h na CPU)
    python main.py --quick   # 10% podataka, 2 epohe (~10 min na CPU)

Dataset struktura:
    data/soccer_events/Dataset/EventClasses/
        goal/, gattempts/, freekick/, corner/,
        offside/, yellowcard/, redc/,
        playercelebration/, plentystock/, spectator/
"""

import json
import random
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")  # bez GUI — čuvamo kao fajlovi
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss: FL(p_t) = -(1 - p_t)^γ · log(p_t)"""

    def __init__(self, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        n_cls = tf.shape(y_pred)[-1]
        y_oh = tf.one_hot(y_true, n_cls)
        pt = tf.reduce_sum(y_oh * y_pred, axis=-1)
        return tf.pow(1.0 - pt, self.gamma) * (-tf.math.log(pt))

    def get_config(self):
        cfg = super().get_config()
        cfg["gamma"] = self.gamma
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURACIJA
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
DATA_ROOT = Path("./data/soccer_events/Dataset/EventClasses")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
    "goal": "goal",
    "gattempts": "goal_attempt",
    "freekick": "free_kick",
    "corner": "free_kick",  # set-piece, spaja se sa slobodnjak
    "offside": "offside",
    "yellowcard": "card",
    "redc": "card",
    "playercelebration": "normal_play",
    "plentystock": "normal_play",
    # 'spectator' isključen
}

LABEL_TO_IDX = {
    "goal": 0,
    "goal_attempt": 1,
    "free_kick": 2,
    "offside": 3,
    "card": 4,
    "normal_play": 5,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(len(LABEL_TO_IDX))]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_P1 = 10
EPOCHS_P2 = 20
LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_TF_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_TF_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# GPU
# ══════════════════════════════════════════════════════════════════════════════


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {len(gpus)} fizički GPU(s)")
    else:
        print("GPU nije dostupan — CPU mod")
    print(f"TensorFlow: {tf.__version__}")


# ══════════════════════════════════════════════════════════════════════════════
# UČITAVANJE DATASETA
# ══════════════════════════════════════════════════════════════════════════════


def load_dataset(data_root: Path) -> tuple:
    """
    Prolazi kroz sve foldere i gradi liste putanja i labela.
    Folderi koji nisu u CLASS_MAPPING se preskaču (spectator).
    """
    image_paths, labels, skipped = [], [], {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset nije pronađen: {data_root}\n"
            "Proveri da je archive.zip raspakovan u data/soccer_events/"
        )

    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if name not in CLASS_MAPPING:
            n = sum(1 for f in folder.iterdir() if f.suffix.lower() in valid_ext)
            skipped[name] = n
            continue

        idx = LABEL_TO_IDX[CLASS_MAPPING[name]]
        for img in folder.iterdir():
            if img.suffix.lower() in valid_ext:
                image_paths.append(str(img))
                labels.append(idx)

    return image_paths, np.array(labels), skipped


def print_distribution(labels_arr):
    print("\nDistribucija klasa:")
    print(f"  {'Klasa':15s}  {'Br. slika':>10s}  {'Procenat':>9s}")
    print("  " + "-" * 38)
    total = len(labels_arr)
    for idx in range(NUM_CLASSES):
        n = int(np.sum(labels_arr == idx))
        print(f"  {IDX_TO_LABEL[idx]:15s}  {n:>10,}  {n / total * 100:>8.1f}%")
    print(f"  {'UKUPNO':15s}  {total:>10,}")


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=10 / 360),
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
        layers.RandomZoom(height_factor=(-0.2, 0.0)),
    ],
    name="augmentation",
)


def preprocess_fn(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    img = (img - _TF_MEAN) / _TF_STD
    return img, label


def build_tf_dataset(paths, labels, is_training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if is_training:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, out, name="soccer_classifier")
    return model, backbone


# ══════════════════════════════════════════════════════════════════════════════
# TRENIRANJE
# ══════════════════════════════════════════════════════════════════════════════


def get_callbacks(phase: int) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3 if phase == 1 else 5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def phase1(model, train_ds, val_ds, class_weights, epochs=EPOCHS_P1):
    print("\n" + "=" * 55)
    print("FAZA 1 — Trening klasifikatora (frozen backbone)")
    print("  Loss: Focal Loss (γ=2.0) — redukuje false positive")
    print("=" * 55)
    model.compile(
        optimizer=keras.optimizers.Adam(LR_PHASE1),
        loss=FocalLoss(gamma=2.0),
        metrics=["accuracy"],
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(1),
    )


def phase2(model, backbone, train_ds, val_ds, class_weights, epochs=EPOCHS_P2):
    print("\n" + "=" * 55)
    print("FAZA 2 — Fine-tuning (block5+ odmrznut)")
    print("  Loss: Focal Loss (γ=2.0) — redukuje false positive")
    print("=" * 55)
    backbone.trainable = True
    started = False
    for layer in backbone.layers:
        if "block5" in layer.name:
            started = True
        layer.trainable = started

    model.compile(
        optimizer=keras.optimizers.Adam(LR_PHASE2),
        loss=FocalLoss(gamma=2.0),
        metrics=["accuracy"],
    )
    print(
        f"Trenabilni parametri: {sum(np.prod(v.shape) for v in model.trainable_variables):,}"
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIJA
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(model, test_ds):
    y_true, y_pred, y_prob = [], [], []
    for imgs, lbls in tqdm(test_ds, desc="Evaluacija"):
        probs = model.predict(imgs, verbose=0)
        y_true.extend(lbls.numpy())
        y_pred.extend(np.argmax(probs, 1))
        y_prob.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    max_conf = y_prob.max(axis=1)
    print("\nDistribucija confidence (max softmax po primeru):")
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        below = int((max_conf < thr).sum())
        pct = below / len(max_conf) * 100
        mask_below = max_conf < thr
        if mask_below.sum() > 0:
            fp_below = int((y_pred[mask_below] != y_true[mask_below]).sum())
            fp_pct = fp_below / mask_below.sum() * 100
        else:
            fp_pct = 0.0
        print(
            f"  conf < {thr:.1f}: {below:4d} primerа ({pct:5.1f}%)  "
            f"od kojih je grešaka: {fp_pct:.1f}%"
        )

    return y_true, y_pred, y_prob


def print_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_p = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    prec = precision_score(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    print("\n" + "=" * 65)
    print("                  REZULTATI NA TEST SETU")
    print("=" * 65)
    print(f"  Top-1 Accuracy:     {acc * 100:6.2f}%  (cilj: >75%)")
    print(f"  Weighted F1-score:  {f1_w:6.4f}  (cilj: >0.72)")
    print("=" * 65)
    print(f"  {'Klasa':15s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
    print("  " + "-" * 42)
    for i, name in IDX_TO_LABEL.items():
        flag = "  ← retka" if name == "offside" else ""
        print(f"  {name:15s}  {prec[i]:6.4f}  {rec[i]:6.4f}  {f1_p[i]:6.4f}{flag}")
    print("=" * 65)
    print(f"\n  Cilj accuracy >75%: {'✓' if acc > 0.75 else '✗'}")
    print(f"  Cilj F1 >0.72:      {'✓' if f1_w > 0.72 else '✗'}")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    path = OUTPUT_DIR / "classification_report.txt"
    with open(path, "w") as f:
        f.write("Soccer Action Recognition\n" + "=" * 55 + "\n")
        f.write(f"Accuracy:    {acc * 100:.2f}%\n")
        f.write(f"Weighted F1: {f1_w:.4f}\n\n")
        f.write(report)
    print(f"\nReport sačuvan: {path}")
    return acc, f1_w, f1_p


# ══════════════════════════════════════════════════════════════════════════════
# VIZUALIZACIJE
# ══════════════════════════════════════════════════════════════════════════════


def plot_training_history(h1, h2):
    p1_n = len(h1.history["loss"])
    loss = h1.history["loss"] + h2.history["loss"]
    vloss = h1.history["val_loss"] + h2.history["val_loss"]
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    ep = list(range(1, len(loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Istorija treniranja", fontsize=14, fontweight="bold")

    for ax, train, val, title, ylabel in [
        (axes[0], loss, vloss, "Loss (Cross-Entropy)", "Loss"),
        (axes[1], acc, vacc, "Accuracy", "Accuracy"),
    ]:
        ax.plot(ep, train, "b-o", ms=4, lw=2, label="Train")
        ax.plot(ep, val, "r-o", ms=4, lw=2, label="Val")
        ax.axvline(p1_n + 0.5, color="gray", ls="--", lw=1.5)
        ax.text(
            p1_n * 0.5,
            max(train) * 0.97,
            "Faza 1",
            ha="center",
            color="gray",
            fontsize=9,
        )
        ax.text(
            p1_n + (len(ep) - p1_n) * 0.5,
            max(train) * 0.97,
            "Faza 2",
            ha="center",
            color="gray",
            fontsize=9,
        )
        if ylabel == "Accuracy":
            ax.axhline(0.75, color="green", ls=":", alpha=0.7, label="Cilj 75%")
            ax.set_ylim(0, 1)
        ax.set_xlabel("Epoha")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "training_history.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, fmt, cmap, title in [
        (axes[0], cm_norm, ".2f", "Blues", "Normalizovana"),
        (axes[1], cm, "d", "Oranges", "Apsolutni brojevi"),
    ]:
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
        )
        ax.set_xlabel("Predviđena", fontsize=11)
        ax.set_ylabel("Stvarna", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")


def plot_per_class_f1(f1_per, f1_w):
    _, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if f >= 0.72 else "#e74c3c" for f in f1_per]
    bars = ax.bar(CLASS_NAMES, f1_per, color=colors, edgecolor="white")
    ax.axhline(0.72, color="navy", ls="--", lw=1.5, label="Cilj 0.72")
    ax.axhline(f1_w, color="orange", lw=1.5, label=f"Weighted avg {f1_w:.3f}")
    for bar, f in zip(bars, f1_per):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{f:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score po klasama", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = OUTPUT_DIR / "per_class_f1.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")


def plot_class_distribution(labels_arr):
    unique, counts = np.unique(labels_arr, return_counts=True)
    names = [IDX_TO_LABEL[i] for i in unique]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribucija klasa", fontsize=13, fontweight="bold")

    bars = axes[0].barh(names, counts, color=colors, edgecolor="white")
    for bar, c in zip(bars, counts):
        axes[0].text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,}",
            va="center",
            fontsize=9,
        )
    axes[0].set_xlabel("Broj slika")
    axes[0].set_title("Apsolutni brojevi")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].pie(
        counts,
        labels=names,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.85,
    )
    axes[1].set_title("Procentualno")

    plt.tight_layout()
    p = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════


class GradCAM:
    def __init__(self, model, backbone):
        target = self._find_target_layer(backbone)
        print(f"GradCAM target: {target.name}")

        self.backbone_grad_model = keras.Model(
            inputs=backbone.inputs, outputs=[target.output, backbone.output]
        )

        # Head slojevi iz kompletnog modela (posle backbone-a)
        self.gap = model.get_layer("global_average_pooling2d")
        self.dense1 = model.get_layer("dense")
        self.dropout = model.get_layer("dropout")
        self.dense2 = model.get_layer("dense_1")

    @staticmethod
    def _find_target_layer(backbone):
        for name in ["top_activation", "top_conv", "block7a_project_bn", "block6d_add"]:
            try:
                return backbone.get_layer(name)
            except ValueError:
                continue
        for layer in reversed(backbone.layers):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except Exception:
                continue
        raise ValueError("Nije pronađen konvolucioni sloj za GradCAM")

    def compute(self, img_batch, class_idx=None):
        img = tf.cast(img_batch, tf.float32)

        with tf.GradientTape() as tape:
            conv_out, backbone_out = self.backbone_grad_model(img, training=False)
            tape.watch(conv_out)

            x = self.gap(backbone_out)
            x = self.dense1(x)
            x = self.dropout(x, training=False)
            preds = self.dense2(x)

            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]

        grads = tape.gradient(score, conv_out)[0]
        alpha = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.maximum(tf.reduce_sum(alpha * conv_out[0], axis=-1), 0)
        cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
        heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return heatmap, class_idx, float(preds[0, class_idx])

    def overlay(self, orig, heatmap, alpha=0.4):
        h = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(np.clip(orig, 0, 1) * (1 - alpha) + h * alpha, 0, 1)


def plot_false_positives(test_paths, y_true, y_pred, y_prob, n_examples=8):
    """
    Svaki primer prikazan kao zasebna kartica:
      - Gornji deo: slika
      - Sredina: STVARNO (zeleno) vs PREDVIĐENO (crveno)
      - Donji deo: bar chart svih 6 klasa sa % sigurnosti
    """
    error_idxs = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if not error_idxs:
        print("Nema grešaka.")
        return

    selected = random.sample(error_idxs, min(n_examples, len(error_idxs)))

    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 5.5))
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(
        "False Positives — Pogrešne predikcije modela",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for plot_i, idx in enumerate(selected):
        row = plot_i // n_cols
        col = plot_i % n_cols
        ax = axes[row][col]
        ax.axis("off")

        true_cls = int(y_true[idx])
        pred_cls = int(y_pred[idx])
        probs = y_prob[idx]
        true_name = IDX_TO_LABEL[true_cls]
        pred_name = IDX_TO_LABEL[pred_cls]

        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_img = ax.inset_axes([0, 0.45, 1, 0.55])
        raw = tf.io.read_file(test_paths[idx])
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), [IMG_SIZE, IMG_SIZE])
        ax_img.imshow(np.clip(img.numpy() / 255.0, 0, 1))
        ax_img.axis("off")

        ax.text(
            0.5,
            0.43,
            f"✓ STVARNO:  {true_name.upper()}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="#27ae60",
                edgecolor="none",
                boxstyle="round,pad=0.3",
                alpha=0.95,
            ),
        )
        ax.text(
            0.5,
            0.35,
            f"✗ MODEL:     {pred_name.upper()}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="#e74c3c",
                edgecolor="none",
                boxstyle="round,pad=0.3",
                alpha=0.95,
            ),
        )

        ax_bar = ax.inset_axes([0.02, 0.0, 0.96, 0.30])
        bar_colors = []
        for ci in range(NUM_CLASSES):
            if ci == true_cls:
                bar_colors.append("#27ae60")
            elif ci == pred_cls:
                bar_colors.append("#e74c3c")
            else:
                bar_colors.append("#d5d8dc")

        y_pos = range(NUM_CLASSES)
        bars = ax_bar.barh(
            y_pos, probs, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7
        )

        for ci, (bar, prob) in enumerate(zip(bars, probs)):
            name = IDX_TO_LABEL[ci]
            label = f" {name}  {prob * 100:.0f}%"
            ax_bar.text(
                0.01,
                ci,
                label,
                va="center",
                ha="left",
                fontsize=7.5,
                fontweight="bold" if ci in (true_cls, pred_cls) else "normal",
                color="white" if ci in (true_cls, pred_cls) else "#333",
                transform=ax_bar.get_yaxis_transform(),
            )

        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0, 0.5, 1.0])
        ax_bar.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
        ax_bar.spines[["top", "right", "left"]].set_visible(False)
        ax_bar.tick_params(axis="x", length=2)

    for plot_i in range(len(selected), n_rows * n_cols):
        axes[plot_i // n_cols][plot_i % n_cols].axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "false_positives.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")

    print("\nNajčešće greške:")
    errors = {}
    for t, p_ in zip(y_true, y_pred):
        if t != p_:
            errors[(int(t), int(p_))] = errors.get((int(t), int(p_)), 0) + 1
    print(f"  {'STVARNO':15s} → {'MODEL KAŽE':15s}  {'Br.':>5s}")
    print("  " + "-" * 40)
    for (t, p_), cnt in sorted(errors.items(), key=lambda x: -x[1])[:10]:
        print(f"  {IDX_TO_LABEL[t]:15s} → {IDX_TO_LABEL[p_]:15s}  {cnt:>5}")


def plot_gradcam(gradcam, test_paths, y_true, y_pred, n=3):
    per_class = {}
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            per_class.setdefault(t, []).append(i)

    n_cols = n * 2
    fig, axes = plt.subplots(NUM_CLASSES, n_cols, figsize=(n_cols * 3, NUM_CLASSES * 3))
    fig.suptitle('Grad-CAM — Šta model "gleda"', fontsize=13, fontweight="bold")

    for row, cls in enumerate(range(NUM_CLASSES)):
        axes[row][0].set_ylabel(
            IDX_TO_LABEL[cls].upper(), fontsize=8, fontweight="bold"
        )
        idxs = random.sample(
            per_class.get(cls, []), min(n, len(per_class.get(cls, [])))
        )

        for col_pair, idx in enumerate(idxs):
            raw = tf.io.read_file(test_paths[idx])
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img_255 = tf.image.resize(
                tf.cast(img, tf.float32), [IMG_SIZE, IMG_SIZE]
            ).numpy()
            img_01 = img_255 / 255.0

            img_norm = (img_01 - IMAGENET_MEAN) / IMAGENET_STD
            heatmap, _, conf = gradcam.compute(img_norm[None], class_idx=cls)
            ov = gradcam.overlay(img_01, heatmap)

            axes[row][col_pair * 2].imshow(np.clip(img_01, 0, 1))
            axes[row][col_pair * 2].axis("off")
            axes[row][col_pair * 2 + 1].imshow(ov)
            axes[row][col_pair * 2 + 1].set_title(
                f"{conf * 100:.0f}%",
                fontsize=8,
                backgroundcolor="#e74c3c",
                color="white",
                pad=1,
            )
            axes[row][col_pair * 2 + 1].axis("off")

        for empty in range(len(idxs), n):
            axes[row][empty * 2].axis("off")
            axes[row][empty * 2 + 1].axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "gradcam_examples.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════════════════


def compute_baseline(y_train, y_test, model_acc, model_f1):
    maj_cls = int(np.bincount(y_train).argmax())
    y_maj = np.full_like(y_test, maj_cls)
    acc_maj = accuracy_score(y_test, y_maj)
    f1_maj = f1_score(y_test, y_maj, average="weighted", zero_division=0)

    rng = np.random.default_rng(SEED)
    probs = np.bincount(y_train) / len(y_train)
    y_rnd = rng.choice(NUM_CLASSES, size=len(y_test), p=probs)
    acc_rnd = accuracy_score(y_test, y_rnd)
    f1_rnd = f1_score(y_test, y_rnd, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("  POREĐENJE SA BASELINE MODELIMA")
    print("=" * 60)
    print(f"  {'Model':30s} {'Accuracy':>9s} {'F1':>8s}")
    print("  " + "-" * 50)
    print(
        f"  {'Majority (' + IDX_TO_LABEL[maj_cls] + ')':30s} {acc_maj * 100:>8.2f}% {f1_maj:>8.4f}"
    )
    print(f"  {'Random':30s} {acc_rnd * 100:>8.2f}% {f1_rnd:>8.4f}")
    print(f"  {'EfficientNetB0 (naš)':30s} {model_acc * 100:>8.2f}% {model_f1:>8.4f}")
    print("=" * 60)

    models = ["Majority\nclass", "Random", "EfficientNetB0\n(naš)"]
    accs = [acc_maj * 100, acc_rnd * 100, model_acc * 100]
    f1s = [f1_maj, f1_rnd, model_f1]
    clrs = ["#95a5a6", "#bdc3c7", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Poređenje modela", fontsize=13, fontweight="bold")
    for ax, vals, ylabel, target, target_lbl in [
        (axes[0], accs, "Accuracy (%)", 75, "Cilj 75%"),
        (axes[1], f1s, "Weighted F1", 0.72, "Cilj 0.72"),
    ]:
        bars = ax.bar(models, vals, color=clrs, edgecolor="white")
        ax.axhline(target, color="red", ls="--", label=target_lbl)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(vals),
                f"{v:.1f}" if ylabel == "Accuracy (%)" else f"{v:.3f}",
                ha="center",
                fontweight="bold",
            )
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = OUTPUT_DIR / "baseline_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sačuvano: {p}")

    return acc_maj, f1_maj, acc_rnd, f1_rnd


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main(quick: bool = False, sample_pct: int = 100):
    setup_gpu()

    if quick:
        pct = 5
        ep1, ep2 = 2, 1
        print("\n⚡ QUICK MOD — 5% podataka, 2+1 epohe (~5 min na CPU)")
    else:
        pct = max(1, min(100, sample_pct))
        ep1, ep2 = EPOCHS_P1, EPOCHS_P2
        if pct < 100:
            print(f"\n📊 SAMPLE MOD — {pct}% podataka, {ep1}/{ep2} epoha")

    # ── 1. Učitavanje ─────────────────────────────────────────────────────────
    print("\n[1/7] Učitavanje dataseta...")
    paths, labels, skipped = load_dataset(DATA_ROOT)
    print(f"Učitano: {len(paths):,} slika")
    if skipped:
        print(f"Preskočeno: {skipped}")

    if pct < 100:
        paths, _, labels, _ = train_test_split(
            paths, labels, train_size=pct / 100, stratify=labels, random_state=SEED
        )
        print(f"Smanjen na {len(paths):,} slika ({pct}%)")

    print_distribution(labels)
    plot_class_distribution(labels)

    # ── 2. Podela ─────────────────────────────────────────────────────────────
    print("\n[2/7] Stratifikovana podela 70/15/15...")
    X_tv, X_test, y_tv, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=SEED
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── 3. Class weights ──────────────────────────────────────────────────────
    cw_raw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_arr = np.sqrt(cw_raw)
    cw_arr = cw_arr / cw_arr.min()
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), cw_arr)}
    print("\nClass weights (sqrt-balanced, min=1.0):")
    print(f"  {'Klasa':15s}  {'raw balanced':>13s}  {'sqrt (korišćeno)':>16s}")
    print("  " + "-" * 48)
    for idx, (raw, adj) in zip(np.unique(y_train), zip(cw_raw, cw_arr)):
        print(f"  {IDX_TO_LABEL[int(idx)]:15s}  {raw:>13.4f}  {adj:>16.4f}")

    # ── 4. Datasetovi ─────────────────────────────────────────────────────────
    print("\n[3/7] Kreiranje tf.data pipeline-a...")
    train_ds = build_tf_dataset(X_train, y_train, is_training=True)
    val_ds = build_tf_dataset(X_val, y_val, is_training=False)
    test_ds = build_tf_dataset(X_test, y_test, is_training=False)

    # ── 5. Model ──────────────────────────────────────────────────────────────
    print("\n[4/7] Izgradnja modela...")
    model, backbone = build_model()
    total = model.count_params()
    train_p = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Ukupno parametara:    {total:,}")
    print(f"  Trenabilni (Faza 1):  {train_p:,}")
    if not quick:
        model.summary(line_length=80)

    # ── 6. Treniranje ─────────────────────────────────────────────────────────
    print("\n[5/7] Treniranje...")
    h1 = phase1(model, train_ds, val_ds, cw_dict, epochs=ep1)
    h2 = phase2(model, backbone, train_ds, val_ds, cw_dict, epochs=ep2)
    plot_training_history(h1, h2)

    # ── 7. Evaluacija ─────────────────────────────────────────────────────────
    print("\n[6/7] Evaluacija...")
    best = keras.models.load_model(
        str(OUTPUT_DIR / "best_model.keras"), custom_objects={"FocalLoss": FocalLoss}
    )
    y_true, y_pred, y_prob = evaluate(best, test_ds)
    acc, f1_w, f1_per = print_results(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_f1(f1_per, f1_w)
    plot_false_positives(X_test, y_true, y_pred, y_prob)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("\n[7/7] Grad-CAM vizualizacija...")
    best_backbone = best.get_layer("efficientnetb0")
    gradcam = GradCAM(best, best_backbone)
    plot_gradcam(gradcam, X_test, y_true, y_pred, n=3)

    # ── Baseline ──────────────────────────────────────────────────────────────
    acc_maj, f1_maj, acc_rnd, f1_rnd = compute_baseline(y_train, y_test, acc, f1_w)

    # ── Rezultati JSON ────────────────────────────────────────────────────────
    results = {
        "model": {
            "backbone": "EfficientNetB0",
            "num_classes": NUM_CLASSES,
            "total_params": int(best.count_params()),
        },
        "dataset": {
            "total": len(paths),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "class_counts": {
                IDX_TO_LABEL[i]: int(np.sum(labels == i)) for i in range(NUM_CLASSES)
            },
        },
        "test_metrics": {
            "accuracy": float(acc),
            "weighted_f1": float(f1_w),
            "per_class_f1": {
                IDX_TO_LABEL[i]: float(f1_per[i]) for i in range(NUM_CLASSES)
            },
            "goal_accuracy_met": acc > 0.75,
            "goal_f1_met": f1_w > 0.72,
        },
        "baseline": {
            "majority_accuracy": float(acc_maj),
            "majority_f1": float(f1_maj),
            "random_accuracy": float(acc_rnd),
            "random_f1": float(f1_rnd),
        },
    }
    rp = OUTPUT_DIR / "results_summary.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSvi rezultati sačuvani u: {OUTPUT_DIR.resolve()}/")
    print(json.dumps(results["test_metrics"], indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="10%% podataka, 2 epohe — brzi test (~10 min)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        metavar="PCT",
        help="Procenat podataka koji se koristi (1-100, default=100). "
        "Preporuka za fakultetski projekat: --sample 25",
    )
    args = parser.parse_args()
    main(quick=args.quick, sample_pct=args.sample)
