"""
model.py  —  DenseNet-based classifier
---------------------------------------
• build_model()   : constructs a DenseNet121 with the top 30 layers unfrozen
• train_model()   : fits the model and returns history + metrics
• evaluate_model(): full evaluation (accuracy, precision, recall, confusion matrix)
• compare_and_save(): compare new model vs all old models on the validation set,
                      save the best one locally + to Google Drive
"""

import json
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend for servers
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from preprocessing import get_data_generators, build_combined_dataset

# ─── Paths & hyper-parameters ─────────────────────────────────────────────────
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

UNFREEZE_TOP_N  = 30       # number of DenseNet layers to unfreeze
IMG_SIZE        = (224, 224, 3)
EPOCHS_FROZEN   = 5        # warm-up with base frozen
EPOCHS_FINETUNE = 15       # fine-tune with top layers unfrozen
LEARNING_RATE   = 1e-4

# Google Drive (set to None to disable GDrive upload)
GDRIVE_MODEL_DIR = "/content/drive/MyDrive/MLPipeline/models"


# ─── Model Construction ───────────────────────────────────────────────────────

def build_model(num_classes: int) -> Model:
    """
    DenseNet121 backbone.
    Phase 1 : all layers frozen   → warm up the new head
    Phase 2 : top-30 unfrozen     → fine-tune
    """
    base = DenseNet121(
        weights    = "imagenet",
        include_top= False,
        input_shape= IMG_SIZE,
    )
    base.trainable = False          # start fully frozen

    # ── Custom classification head ────────────────────────────────────────────
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    print(f"[Model] Built DenseNet121 — {num_classes} classes, "
          f"total layers: {len(model.layers)}")
    return model


def _unfreeze_top(model: Model, n: int = UNFREEZE_TOP_N) -> None:
    """Unfreeze the last *n* layers of the backbone."""
    base = model.layers[0]          # DenseNet121 is the first layer
    for layer in base.layers[-n:]:
        layer.trainable = True
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"[Model] Unfroze top-{n} backbone layers — "
          f"{trainable} trainable layers total")


def _compile(model: Model, lr: float = LEARNING_RATE) -> None:
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
    )


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    train_gen,
    val_gen,
    class_names: list[str],
    model_name: str | None = None,
    existing_model: Model | None = None,
) -> tuple[Model, dict]:
    """
    Full two-phase training (frozen warm-up → fine-tune).

    Parameters
    ----------
    existing_model : Model | None
        If provided, skip construction and use this model as the starting
        point (useful for retraining scenarios).

    Returns
    -------
    (model, history_dict)
    """
    num_classes = len(class_names)
    model = existing_model if existing_model else build_model(num_classes)

    if model_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"densenet_{ts}"

    log_dir = MODELS_DIR / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=str(log_dir / "best_weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, verbose=1),
    ]

    # ── Phase 1: frozen backbone ───────────────────────────────────────────
    print("\n[Model] ── Phase 1: Warm-up (backbone frozen) ──")
    _compile(model, lr=1e-3)
    h1 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS_FROZEN,
        callbacks       = cb,
        verbose         = 1,
    )

    # ── Phase 2: fine-tune top-30 layers ──────────────────────────────────
    print(f"\n[Model] ── Phase 2: Fine-tune (top-{UNFREEZE_TOP_N} unfrozen) ──")
    _unfreeze_top(model, UNFREEZE_TOP_N)
    _compile(model, lr=LEARNING_RATE)
    h2 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS_FINETUNE,
        callbacks       = cb,
        verbose         = 1,
        initial_epoch   = len(h1.history["loss"]),
    )

    # ── Merge histories ────────────────────────────────────────────────────
    history = {}
    for k in h1.history:
        history[k] = h1.history[k] + h2.history[k]

    # ── Save training curves ───────────────────────────────────────────────
    _plot_history(history, log_dir)

    # ── Save model ────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"{model_name}.h5"
    model.save(str(model_path))
    print(f"[Model] Saved → {model_path}")

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "model_name" : model_name,
        "classes"    : class_names,
        "num_classes": num_classes,
        "timestamp"  : datetime.now().isoformat(),
        "final_val_accuracy": float(history["val_accuracy"][-1]),
    }
    with open(MODELS_DIR / f"{model_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return model, history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model: Model, val_gen, class_names: list[str]) -> dict:
    """
    Return a comprehensive metrics dict:
    accuracy, precision, recall, f1, confusion_matrix
    """
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes[: len(y_pred)]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy"        : round(acc,  4),
        "precision"       : round(prec, 4),
        "recall"          : round(rec,  4),
        "f1_score"        : round(f1,   4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    print(f"[Eval] Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")
    return metrics


# ─── Compare & Save ───────────────────────────────────────────────────────────

def compare_and_save(
    new_model     : Model,
    new_metrics   : dict,
    model_name    : str,
    val_gen,
    class_names   : list[str],
) -> str:
    """
    Compare the newly trained model against every existing .h5 in models/.
    Save the best one and upload to Google Drive (if mounted).

    Returns the name of the winning model.
    """
    best_acc  = new_metrics["accuracy"]
    best_name = model_name

    # ── Load & evaluate every existing model ──────────────────────────────
    for h5 in MODELS_DIR.glob("*.h5"):
        if h5.stem == model_name:
            continue
        try:
            old_model   = tf.keras.models.load_model(str(h5))
            old_metrics = evaluate_model(old_model, val_gen, class_names)
            print(f"[Compare] {h5.stem}: accuracy={old_metrics['accuracy']:.4f}")
            if old_metrics["accuracy"] > best_acc:
                best_acc  = old_metrics["accuracy"]
                best_name = h5.stem
        except Exception as exc:
            print(f"[Compare] Could not evaluate {h5.stem}: {exc}")

    print(f"[Compare] Best model: {best_name}  (accuracy={best_acc:.4f})")

    # ── Always persist the new model ──────────────────────────────────────
    final_path = MODELS_DIR / f"{model_name}.h5"
    new_model.save(str(final_path))

    # ── Upload to Google Drive if available ───────────────────────────────
    _upload_to_gdrive(final_path)

    return best_name


def _upload_to_gdrive(model_path: Path) -> None:
    """Copy model file to Google Drive (only works in Colab with Drive mounted)."""
    if GDRIVE_MODEL_DIR is None:
        return
    gdrive = Path(GDRIVE_MODEL_DIR)
    if gdrive.exists():
        dest = gdrive / model_path.name
        shutil.copy2(model_path, dest)
        print(f"[GDrive] Uploaded → {dest}")
    else:
        print("[GDrive] Drive not mounted — skipping upload.")


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_history(history: dict, save_dir: Path) -> None:
    """Save accuracy and loss curves as PNG files."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history["accuracy"],     label="Train Accuracy",     color="#4C9BE8")
    ax1.plot(history["val_accuracy"], label="Val Accuracy",       color="#E87C4C")
    ax1.set_title("Accuracy over Epochs", fontsize=14)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history["loss"],     label="Train Loss", color="#4C9BE8")
    ax2.plot(history["val_loss"], label="Val Loss",   color="#E87C4C")
    ax2.set_title("Loss over Epochs", fontsize=14)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"[Model] Training curves saved → {save_dir / 'training_curves.png'}")


# ─── Entry point for full training run ───────────────────────────────────────

if __name__ == "__main__":
    combined   = build_combined_dataset(new_data_path=None)
    train_gen, val_gen, class_names = get_data_generators(combined)

    model, history = train_model(train_gen, val_gen, class_names)
    metrics        = evaluate_model(model, val_gen, class_names)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "classification_report"}, indent=2))
