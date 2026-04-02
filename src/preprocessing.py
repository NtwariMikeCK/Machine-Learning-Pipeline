"""
preprocessing.py
----------------
Handles all data preprocessing for training and retraining.
- Combines old data (train + test) with new incoming data
- Applies augmentation via ImageDataGenerator
- Returns 80/20 train/validation split in batches
"""

import os
import shutil
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # DenseNet expects 224×224
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
OLD_TRAIN   = Path("old_data/train")
OLD_TEST    = Path("old_data/test")
COMBINED    = Path("data/combined")   # temporary merged directory


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _collect_classes(directory: Path) -> set:
    """Return the set of class-folder names inside *directory*."""
    return {p.name for p in directory.iterdir() if p.is_dir()}


def _merge_into(src: Path, dst: Path) -> None:
    """
    Copy every image file from src/<class>/ into dst/<class>/.
    Creates the destination class-folders if they don't exist yet.
    """
    if not src.exists():
        return
    for cls_dir in src.iterdir():
        if not cls_dir.is_dir():
            continue
        target = dst / cls_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(img, target / img.name)


def _copy_new_data(new_data_root: Path, dst: Path) -> None:
    """
    Copy newly-uploaded images (organised as new_data_root/<class>/<images>)
    into the combined directory.
    """
    _merge_into(new_data_root, dst)


# ─── Public API ───────────────────────────────────────────────────────────────

def build_combined_dataset(new_data_path: str | None = None) -> Path:
    """
    Merge old_data/train + old_data/test (+ optional new data) into
    data/combined/ and return that path.

    Parameters
    ----------
    new_data_path : str | None
        Path to the folder that holds the newly-uploaded images
        (expected layout: <new_data_path>/<class_name>/<image_files>).
        Pass None to skip adding new data (initial training).

    Returns
    -------
    Path  pointing to the combined dataset directory.
    """
    # Wipe and recreate the combined directory so we start clean
    if COMBINED.exists():
        shutil.rmtree(COMBINED)
    COMBINED.mkdir(parents=True)

    # 1️⃣  Merge old training split
    print("[Preprocessing] Merging old_data/train …")
    _merge_into(OLD_TRAIN, COMBINED)

    # 2️⃣  Merge old test split (we want all labelled data for retraining)
    print("[Preprocessing] Merging old_data/test …")
    _merge_into(OLD_TEST, COMBINED)

    # 3️⃣  Merge newly uploaded data (if provided)
    if new_data_path:
        new_root = Path(new_data_path)
        if new_root.exists():
            print(f"[Preprocessing] Merging new data from {new_root} …")
            _copy_new_data(new_root, COMBINED)
        else:
            print(f"[Preprocessing] Warning: new_data_path '{new_data_path}' does not exist, skipping.")

    classes = _collect_classes(COMBINED)
    print(f"[Preprocessing] Combined dataset ready — classes: {sorted(classes)}")
    return COMBINED


def get_data_generators(dataset_path: Path | None = None):
    """
    Build Keras ImageDataGenerators for training and validation.

    Parameters
    ----------
    dataset_path : Path | None
        Root of the combined dataset.  If None, build_combined_dataset()
        is called without new data (initial training from old_data only).

    Returns
    -------
    train_gen, val_gen, class_names : (DirectoryIterator, DirectoryIterator, list[str])
    """
    if dataset_path is None:
        dataset_path = build_combined_dataset()

    # ── Augmentation for the training split ──────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale            = 1.0 / 255,
        validation_split   = VAL_SPLIT,
        rotation_range     = 20,
        width_shift_range  = 0.15,
        height_shift_range = 0.15,
        shear_range        = 0.1,
        zoom_range         = 0.2,
        horizontal_flip    = True,
        fill_mode          = "nearest",
    )

    # ── No augmentation for the validation split ─────────────────────────────
    val_datagen = ImageDataGenerator(
        rescale          = 1.0 / 255,
        validation_split = VAL_SPLIT,
    )

    common_kwargs = dict(
        directory   = str(dataset_path),
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        shuffle     = True,
        seed        = 42,
    )

    train_gen = train_datagen.flow_from_directory(subset="training",   **common_kwargs)
    val_gen   = val_datagen.flow_from_directory(subset="validation", **common_kwargs)

    class_names = list(train_gen.class_indices.keys())
    print(f"[Preprocessing] Train samples : {train_gen.samples}")
    print(f"[Preprocessing] Val   samples : {val_gen.samples}")
    print(f"[Preprocessing] Classes       : {class_names}")

    return train_gen, val_gen, class_names


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    combined = build_combined_dataset()
    train_gen, val_gen, classes = get_data_generators(combined)
    batch_x, batch_y = next(iter(train_gen))
    print(f"Batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
