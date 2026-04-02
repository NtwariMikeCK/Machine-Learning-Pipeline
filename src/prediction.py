"""
prediction.py  —  FastAPI application
--------------------------------------
API 1  POST /predict              – upload an image, get a prediction
API 2  POST /retrain              – upload new labelled images, retrain model
API 3  GET  /system/stats         – API + model performance stats
API 4  GET  /models               – list available models (local + GDrive)
API 5  GET  /health               – liveness probe
API 6  GET  /metrics/{model_name} – saved metrics for a specific model
API 7  POST /gdrive/upload-data   – upload a new-data ZIP to Google Drive
API 8  POST /gdrive/sync-models   – pull latest models from Google Drive

Run locally:
    uvicorn prediction:app --host 0.0.0.0 --port 8000 --reload
"""
from dotenv import load_dotenv
import io
import os
import re
import sys
import json
import time
import shutil
import zipfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import tensorflow as tf

from system import (
    record_api_call, get_api_stats, health_check,
    list_available_models, get_model_metrics, measure_inference,
)

from gdrive import (
    gdrive_available,
    gdrive_status,
    sync_models_from_gdrive,
    upload_data_zip,
    list_gdrive_models,
)


# ─── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR       = Path("models")
UPLOAD_DIR       = Path("data/uploads")
GDRIVE_MODEL_DIR = Path("/content/drive/MyDrive/MLPipeline/models")  # Colab
IMG_SIZE         = (224, 224)

MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ML Pipeline API",
    description = "DenseNet image classifier – prediction, retraining, monitoring",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Model Cache ──────────────────────────────────────────────────────────────
_model_cache: dict[str, tf.keras.Model] = {}
_meta_cache:  dict[str, dict]           = {}


# ─── Keras Version Compatibility Patch ────────────────────────────────────────

_INCOMPATIBLE_LAYER_KEYS = [
    "quantization_config",
]

_STRIP_PATTERNS = [
    re.compile(
        r',?\s*"' + re.escape(key) + r'":\s*(?:null|true|false|"[^"]*"|-?\d+(?:\.\d+)?|\{[^}]*\})',
        re.DOTALL,
    )
    for key in _INCOMPATIBLE_LAYER_KEYS
]


def _patch_keras_file(src: Path) -> Path:
    """
    Return a path to a version-compatible copy of *src* (.keras file).
    Strips keys listed in _INCOMPATIBLE_LAYER_KEYS from config.json.
    """
    patched_path = src.parent / f"{src.stem}_patched.keras"

    if (
        patched_path.exists()
        and patched_path.stat().st_mtime >= src.stat().st_mtime
    ):
        print(f"[PATCH] Using cached patched model: {patched_path.name}")
        return patched_path

    tmp_dir = src.parent / f"_patch_tmp_{src.stem}"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmp_dir)

        config_file = tmp_dir / "config.json"
        if not config_file.exists():
            print(f"[PATCH] No config.json found in {src.name} — skipping patch")
            return src

        original_text = config_file.read_text(encoding="utf-8")
        patched_text  = original_text
        keys_removed  = []

        for key, pattern in zip(_INCOMPATIBLE_LAYER_KEYS, _STRIP_PATTERNS):
            new_text, n = pattern.subn("", patched_text)
            if n:
                patched_text = new_text
                keys_removed.append(f"{key} ({n} occurrence{'s' if n > 1 else ''})")

        if not keys_removed:
            print(f"[PATCH] No incompatible keys found in {src.name} — no patch needed")
            return src

        config_file.write_text(patched_text, encoding="utf-8")
        print(f"[PATCH] Removed from {src.name}: {', '.join(keys_removed)}")

        with zipfile.ZipFile(patched_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(tmp_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(tmp_dir))

        print(f"[PATCH] Patched model saved → {patched_path.name}")
        return patched_path

    except Exception as exc:
        print(f"[PATCH] WARNING: patching failed ({exc}); loading original")
        return src

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


# ─── Model Loading ────────────────────────────────────────────────────────────

def _load_model(model_name: str) -> tf.keras.Model:
    """
    Load (and cache) a model by name.
    Tries local models/ first; if not found, attempts a GDrive sync then retries.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    # ── Prefer .keras ─────────────────────────────────────────────────────────
    local_path_keras = MODELS_DIR / f"{model_name}.keras"
    if local_path_keras.exists():
        load_path = _patch_keras_file(local_path_keras)
        model = tf.keras.models.load_model(str(load_path))
        _model_cache[model_name] = model
        print(f"[API] Loaded model from {load_path}")
        return model

    # ── Fallback to .h5 ───────────────────────────────────────────────────────
    local_path_h5 = MODELS_DIR / f"{model_name}.h5"
    if local_path_h5.exists():
        model = tf.keras.models.load_model(str(local_path_h5), compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        _model_cache[model_name] = model
        print(f"[API] Loaded model from {local_path_h5}")
        return model

    # ── Not found locally — try pulling from Google Drive ────────────────────
    print(f"[API] Model '{model_name}' not found locally. Attempting GDrive sync …")
    sync_models_from_gdrive()

    # Retry after sync
    if local_path_keras.exists():
        load_path = _patch_keras_file(local_path_keras)
        model = tf.keras.models.load_model(str(load_path))
        _model_cache[model_name] = model
        print(f"[API] Loaded model from {load_path} (after GDrive sync)")
        return model

    if local_path_h5.exists():
        model = tf.keras.models.load_model(str(local_path_h5), compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        _model_cache[model_name] = model
        print(f"[API] Loaded model from {local_path_h5} (after GDrive sync)")
        return model

    raise FileNotFoundError(f"Model '{model_name}' not found locally or on Google Drive.")


def _get_latest_model_name() -> str:
    """Return the stem of the newest .keras or .h5 in models/."""
    all_files = sorted(
        list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
        key=lambda p: p.stat().st_mtime,
    )
    # Exclude patched copies
    all_files = [f for f in all_files if "_patched" not in f.stem]
    if not all_files:
        raise FileNotFoundError("No trained models found in models/ directory.")
    return all_files[-1].stem


def _get_class_names(model_name: str) -> list[str]:
    meta_path = MODELS_DIR / f"{model_name}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f).get("classes", [])
    return []


def _preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)          # (1, 224, 224, 3)


def _save_prediction_image(file_bytes: bytes, predicted_class: str) -> None:
    """Persist prediction images for future retraining data collection."""
    save_dir = Path("data/prediction_log") / predicted_class
    save_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = save_dir / f"pred_{ts}.jpg"
    with open(dst, "wb") as f:
        f.write(file_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Startup  —  sync models from Google Drive when the server starts
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    print("[STARTUP] Syncing models from Google Drive …")
    try:
        downloaded = sync_models_from_gdrive()
        if downloaded:
            print(f"[STARTUP] Downloaded models: {downloaded}")
        else:
            print("[STARTUP] No new models to download (or Drive not configured).")
    except Exception as exc:
        print(f"[STARTUP] GDrive sync failed: {exc}")
    print("[STARTUP] Startup complete.")


# ─────────────────────────────────────────────────────────────────────────────
# API 1  –  Predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file       : UploadFile      = File(..., description="Image file to classify"),
    model_name : Optional[str]   = Form(None, description="Model name to use (omit for latest)"),
    save_image : bool            = Form(True, description="Whether to save the image for future retraining"),
):
    """
    Classify an uploaded image.

    - **file**       : JPEG / PNG image
    - **model_name** : which saved model to use (default: latest)
    - **save_image** : persist image to prediction_log/ for future retraining
    """
    t_start = time.perf_counter()
    try:
        name        = model_name or _get_latest_model_name()
        model       = _load_model(name)
        class_names = _get_class_names(name)

        contents = await file.read()
        img_arr  = _preprocess_image(contents)

        t_inf  = time.perf_counter()
        preds  = model.predict(img_arr, verbose=0)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        class_idx  = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        label      = class_names[class_idx] if class_names else str(class_idx)

        conf_map = {
            (class_names[i] if class_names else str(i)): round(float(p), 4)
            for i, p in enumerate(preds[0].tolist())
        }

        if save_image:
            _save_prediction_image(contents, label)

        total_ms = (time.perf_counter() - t_start) * 1000
        record_api_call("/predict", total_ms, 200, inf_ms)

        return {
            "predicted_class"   : label,
            "confidence"        : round(confidence, 4),
            "all_probabilities" : conf_map,
            "model_used"        : name,
            "inference_time_ms" : round(inf_ms, 3),
            "total_latency_ms"  : round(total_ms, 3),
        }

    except FileNotFoundError as exc:
        ms = (time.perf_counter() - t_start) * 1000
        record_api_call("/predict", ms, 404)
        raise HTTPException(404, str(exc))
    except Exception as exc:
        ms = (time.perf_counter() - t_start) * 1000
        record_api_call("/predict", ms, 500)
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# API 2  –  Retrain  (cloud / GPU mode only)
# ─────────────────────────────────────────────────────────────────────────────

_retrain_status = {"running": False, "last": None}


def _do_retrain(zip_path: Path):
    """Background task: extract → preprocess → train → evaluate → save → upload to Drive."""
    _retrain_status["running"] = True
    _retrain_status["started"] = datetime.now().isoformat()
    try:
        extract_root = UPLOAD_DIR / "extracted"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        from preprocessing import build_combined_dataset, get_data_generators
        from model import train_model, evaluate_model, compare_and_save

        combined = build_combined_dataset(new_data_path=str(extract_root))
        train_gen, val_gen, class_names = get_data_generators(combined)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"densenet_retrain_{ts}"

        model, history = train_model(train_gen, val_gen, class_names, model_name=name)
        metrics        = evaluate_model(model, val_gen, class_names)

        metrics_path = MODELS_DIR / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        best = compare_and_save(model, metrics, name, val_gen, class_names)

        # Upload the newly trained model + sidecars to Google Drive
        for suffix in [".h5", ".keras", "_meta.json", "_metrics.json"]:
            candidate = MODELS_DIR / f"{name}{suffix}"
            if candidate.exists():
                from gdrive import upload_model_to_gdrive
                upload_model_to_gdrive(candidate)

        _model_cache.clear()

        _retrain_status.update({
            "running"    : False,
            "last"       : datetime.now().isoformat(),
            "model_name" : name,
            "best_model" : best,
            "metrics"    : {k: v for k, v in metrics.items()
                            if k not in ("confusion_matrix", "classification_report")},
            "status"     : "success",
        })
        print(f"[Retrain] Done – new model: {name}, best overall: {best}")

    except Exception as exc:
        traceback.print_exc()
        _retrain_status.update({
            "running": False,
            "status" : "error",
            "error"  : str(exc),
        })
    finally:
        if zip_path.exists():
            zip_path.unlink()


@app.post("/retrain")
async def retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP archive: <class_name>/<image_files>"),
):
    """
    Upload new labelled images (as a ZIP) and trigger background retraining.
    (Cloud / GPU mode.)  For local mode use /gdrive/upload-data instead.
    """
    if _retrain_status.get("running"):
        raise HTTPException(409, "A retraining job is already running.")

    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip archives are accepted.")

    zip_path = UPLOAD_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    contents = await file.read()
    with open(zip_path, "wb") as f:
        f.write(contents)

    background_tasks.add_task(_do_retrain, zip_path)

    return {"message": "Retraining started in the background.", "poll": "/retrain/status"}


@app.get("/retrain/status")
def retrain_status():
    """Poll the status of the most recent retraining job."""
    return _retrain_status


# ─────────────────────────────────────────────────────────────────────────────
# API 3  –  System / Performance Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/system/stats")
def system_stats(last_n: int = 100):
    """Return rolling API + model performance statistics."""
    t_start = time.perf_counter()
    stats   = get_api_stats(last_n)
    ms      = (time.perf_counter() - t_start) * 1000
    record_api_call("/system/stats", ms, 200)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# API 4  –  List Models  (local + GDrive)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models")
def available_models():
    """Return all available model files — local disk and Google Drive."""
    t_start = time.perf_counter()
    local   = list_available_models()
    gdrive  = list_gdrive_models()
    ms      = (time.perf_counter() - t_start) * 1000
    record_api_call("/models", ms, 200)
    print("Local models:", local)
    print("GDrive models:", gdrive)
    return {"local": local, "gdrive": gdrive}


# ─────────────────────────────────────────────────────────────────────────────
# API 5  –  Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    base = health_check()
    base["gdrive"] = gdrive_status()
    return base


# ─────────────────────────────────────────────────────────────────────────────
# API 6  –  Model Metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics/{model_name}")
def model_metrics(model_name: str):
    """Return saved evaluation metrics for a specific model."""
    data = get_model_metrics(model_name)
    if data is None:
        raise HTTPException(404, f"No metrics found for model '{model_name}'")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# API 7  –  Upload new-data ZIP to Google Drive  (local / Colab mode)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/gdrive/upload-data")
async def gdrive_upload_data(
    file: UploadFile = File(..., description="ZIP of new labelled images to send to Google Drive"),
):
    """
    Save the uploaded ZIP to Google Drive's new-data folder so the Colab
    retraining notebook can pick it up.

    Expected ZIP layout::

        data.zip
        ├── class_a/
        │   └── img001.jpg
        └── class_b/
            └── img001.jpg

    After uploading, open the Colab retrain notebook and run all cells.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip archives are accepted.")

    if not gdrive_available():
        raise HTTPException(503, (
            "Google Drive is not configured on this server. "
            "Set GDRIVE_MODELS_FOLDER_ID, GDRIVE_NEWDATA_FOLDER_ID, and "
            "place the service-account JSON at the configured path."
        ))

    # Save locally first, then stream to Drive
    tmp_path = UPLOAD_DIR / f"newdata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        file_id = upload_data_zip(tmp_path)
        if file_id is None:
            raise HTTPException(500, "Upload to Google Drive failed — check server logs.")
        return {
            "message"    : "ZIP uploaded to Google Drive successfully.",
            "gdrive_id"  : file_id,
            "filename"   : file.filename,
            "size_mb"    : round(len(contents) / 1e6, 2),
            "next_step"  : "Open the Colab retrain notebook and run all cells.",
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# API 8  –  Sync models from Google Drive
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/gdrive/sync-models")
def gdrive_sync_models():
    """
    Pull the latest .keras / .h5 models (and their sidecar JSON files) from
    Google Drive into the local models/ directory.

    Call this after the Colab retraining notebook finishes to make the
    new model available for predictions without restarting the server.
    """
    if not gdrive_available():
        raise HTTPException(503, (
            "Google Drive is not configured on this server."
        ))

    try:
        _model_cache.clear()          # evict stale cached models
        downloaded = sync_models_from_gdrive()
        return {
            "message"    : f"Sync complete. {len(downloaded)} model(s) downloaded.",
            "downloaded" : downloaded,
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"GDrive sync failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = Path("static/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "ML Pipeline API running. See /docs for Swagger UI."}


# ─── Dev server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)