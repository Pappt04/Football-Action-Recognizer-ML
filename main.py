"""
Automatsko prepoznavanje sportskih akcija iz slika
Tim: Stefan Paunović (RA 27/2022) i Tamás Papp (RA 4/2022)
Asistent: Teodor Vidaković

Pokretanje:
    source venv/bin/activate
    python main.py           # puno treniranje (~5-7h na CPU)
    python main.py --quick   # 5% podataka, 2+1 epohe (~5 min na CPU)

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
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from torchvision.models import EfficientNet_B0_Weights

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

from dataset import FootballDataset, get_train_transforms, get_val_transforms
from model import FootballActionRecognizer

warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -(1 - p_t)^γ · log(p_t)"""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        y_oh = F.one_hot(targets, num_classes=probs.shape[1]).float()
        pt = (y_oh * probs).sum(dim=1)

        loss = (1.0 - pt) ** self.gamma * (-torch.log(pt))

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()


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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# GPU
# ══════════════════════════════════════════════════════════════════════════════


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n = torch.cuda.device_count()
        print(f"GPU: {n} fizički GPU(s) — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU nije dostupan — CPU mod")
    print(f"PyTorch: {torch.__version__}")
    return device


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


def build_dataloader(paths, labels, is_training: bool) -> DataLoader:
    transform = get_train_transforms() if is_training else get_val_transforms()
    dataset = FootballDataset(paths, list(labels), transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_training,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRENIRANJE
# ══════════════════════════════════════════════════════════════════════════════


def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    """Run one epoch; returns (avg_loss, accuracy)."""
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = torch.tensor(lbls, dtype=torch.long).to(device) if not isinstance(lbls, torch.Tensor) else lbls.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, lbls)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += len(lbls)

    return total_loss / total, correct / total


def _train_phase(model, train_loader, val_loader, criterion, optimizer, device,
                 epochs, patience, phase):
    """Generic training loop with early stopping + ReduceLROnPlateau."""
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )
    best_val_loss = float("inf")
    no_improve = 0
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoha {epoch:3d}/{epochs} — "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
            f"lr: {lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), str(OUTPUT_DIR / "best_model.pt"))
            print(f"    ✓ Novi best model sačuvan (val_loss: {val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping (patience={patience})")
                break

    # Restore best weights
    model.load_state_dict(
        torch.load(str(OUTPUT_DIR / "best_model.pt"), weights_only=True)
    )
    return history


def phase1(model, train_loader, val_loader, criterion, device, epochs=EPOCHS_P1):
    print("\n" + "=" * 55)
    print("FAZA 1 — Trening klasifikatora (frozen backbone)")
    print("  Loss: Focal Loss (γ=2.0) — redukuje false positive")
    print("=" * 55)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1
    )
    return _train_phase(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs, patience=3, phase=1
    )


def phase2(model, train_loader, val_loader, criterion, device, epochs=EPOCHS_P2):
    print("\n" + "=" * 55)
    print("FAZA 2 — Fine-tuning (block5+ odmrznut)")
    print("  Loss: Focal Loss (γ=2.0) — redukuje false positive")
    print("=" * 55)
    model.unfreeze_from_block5()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trenabilni parametri: {trainable:,}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
    )
    return _train_phase(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs, patience=5, phase=2
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIJA
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Evaluacija"):
            imgs = imgs.to(device)
            lbls = lbls.to(device) if isinstance(lbls, torch.Tensor) else torch.tensor(lbls).to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            y_true.extend(lbls.cpu().numpy())
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
    p1_n = len(h1["loss"])
    loss = h1["loss"] + h2["loss"]
    vloss = h1["val_loss"] + h2["val_loss"]
    acc = h1["accuracy"] + h2["accuracy"]
    vacc = h1["val_accuracy"] + h2["val_accuracy"]
    ep = list(range(1, len(loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Istorija treniranja", fontsize=14, fontweight="bold")

    for ax, train, val, title, ylabel in [
        (axes[0], loss, vloss, "Loss (Focal Loss)", "Loss"),
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
    """
    Grad-CAM using forward/backward hooks on the last conv block of EfficientNet
    (model.features[-1], i.e. features[8] = top Conv2dNormActivation).
    """

    def __init__(self, model: FootballActionRecognizer):
        self.model = model
        self._activations = None
        self._gradients = None

        target_layer = model.features[-1]
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def compute(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        img_tensor: (1, C, H, W) normalized tensor on the correct device.
        Returns: (heatmap np.ndarray H×W in [0,1], class_idx, confidence float)
        """
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(False)

        # Forward
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs[0]))

        # Backward
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-CAM
        acts = self._activations[0]   # (C, H, W)
        grads = self._gradients[0]    # (C, H, W)

        alpha = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = torch.relu((alpha * acts).sum(dim=0))   # (H, W)
        cam = cam.cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        return heatmap, class_idx, float(probs[0, class_idx].detach())

    def overlay(self, orig: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
        h = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(np.clip(orig, 0, 1) * (1 - alpha) + h * alpha, 0, 1)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _load_image_numpy(path: str) -> np.ndarray:
    """Load image as HWC float32 array in [0, 1] at IMG_SIZE×IMG_SIZE."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


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

        ax_img = ax.inset_axes([0, 0.45, 1, 0.55])
        img_01 = _load_image_numpy(test_paths[idx])
        ax_img.imshow(np.clip(img_01, 0, 1))
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


def plot_gradcam(gradcam: GradCAM, test_paths, y_true, y_pred, device, n=3):
    per_class = {}
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            per_class.setdefault(t, []).append(i)

    n_cols = n * 2
    fig, axes = plt.subplots(NUM_CLASSES, n_cols, figsize=(n_cols * 3, NUM_CLASSES * 3))
    fig.suptitle('Grad-CAM — Šta model "gleda"', fontsize=13, fontweight="bold")

    val_transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )

    for row, cls in enumerate(range(NUM_CLASSES)):
        axes[row][0].set_ylabel(
            IDX_TO_LABEL[cls].upper(), fontsize=8, fontweight="bold"
        )
        idxs = random.sample(
            per_class.get(cls, []), min(n, len(per_class.get(cls, [])))
        )

        for col_pair, idx in enumerate(idxs):
            img_01 = _load_image_numpy(test_paths[idx])
            pil_img = Image.fromarray(np.uint8(img_01 * 255))
            img_tensor = val_transform(pil_img).unsqueeze(0).to(device)

            heatmap, _, conf = gradcam.compute(img_tensor, class_idx=cls)
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

    models_labels = ["Majority\nclass", "Random", "EfficientNetB0\n(naš)"]
    accs = [acc_maj * 100, acc_rnd * 100, model_acc * 100]
    f1s = [f1_maj, f1_rnd, model_f1]
    clrs = ["#95a5a6", "#bdc3c7", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Poređenje modela", fontsize=13, fontweight="bold")
    for ax, vals, ylabel, target, target_lbl in [
        (axes[0], accs, "Accuracy (%)", 75, "Cilj 75%"),
        (axes[1], f1s, "Weighted F1", 0.72, "Cilj 0.72"),
    ]:
        bars = ax.bar(models_labels, vals, color=clrs, edgecolor="white")
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
    device = setup_device()

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

    cw_tensor = torch.tensor(
        [cw_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float32
    ).to(device)

    # ── 4. DataLoaderi ────────────────────────────────────────────────────────
    print("\n[3/7] Kreiranje DataLoader pipeline-a...")
    train_loader = build_dataloader(X_train, y_train, is_training=True)
    val_loader = build_dataloader(X_val, y_val, is_training=False)
    test_loader = build_dataloader(X_test, y_test, is_training=False)

    # ── 5. Model ──────────────────────────────────────────────────────────────
    print("\n[4/7] Izgradnja modela...")
    model = FootballActionRecognizer(num_classes=NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Ukupno parametara:    {total:,}")
    print(f"  Trenabilni (Faza 1):  {train_p:,}")

    criterion = FocalLoss(gamma=2.0, weight=cw_tensor)

    # ── 6. Treniranje ─────────────────────────────────────────────────────────
    print("\n[5/7] Treniranje...")
    h1 = phase1(model, train_loader, val_loader, criterion, device, epochs=ep1)
    h2 = phase2(model, train_loader, val_loader, criterion, device, epochs=ep2)
    plot_training_history(h1, h2)

    # ── 7. Evaluacija ─────────────────────────────────────────────────────────
    print("\n[6/7] Evaluacija...")
    best = FootballActionRecognizer(num_classes=NUM_CLASSES)
    best.load_state_dict(
        torch.load(str(OUTPUT_DIR / "best_model.pt"), weights_only=True)
    )
    best = best.to(device)
    best.eval()

    y_true, y_pred, y_prob = evaluate(best, test_loader, device)
    acc, f1_w, f1_per = print_results(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_f1(f1_per, f1_w)
    plot_false_positives(X_test, y_true, y_pred, y_prob)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("\n[7/7] Grad-CAM vizualizacija...")
    gradcam = GradCAM(best)
    plot_gradcam(gradcam, X_test, y_true, y_pred, device, n=3)
    gradcam.remove_hooks()

    # ── Baseline ──────────────────────────────────────────────────────────────
    acc_maj, f1_maj, acc_rnd, f1_rnd = compute_baseline(y_train, y_test, acc, f1_w)

    # ── Rezultati JSON ────────────────────────────────────────────────────────
    results = {
        "model": {
            "backbone": "EfficientNetB0",
            "num_classes": NUM_CLASSES,
            "total_params": int(sum(p.numel() for p in best.parameters())),
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
        help="5%% podataka, 2+1 epohe — brzi test (~5 min)",
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
