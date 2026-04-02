"""
system.py
---------
Monitors both MODEL performance (inference latency) and API performance
(end-to-end request latency, throughput).

Functions exposed
─────────────────
measure_inference()  – time a single model forward pass
get_model_metrics()  – load saved metrics from disk
record_api_call()    – push an API timing record into an in-memory ring buffer
get_api_stats()      – rolling stats over the last N API calls
health_check()       – simple ping/uptime info
"""

import time
import json
import statistics
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR     = Path("models")
RING_BUFFER    = 500          # keep last N API call timings
IMG_SIZE       = (224, 224)
_START_TIME    = time.time()

# In-memory ring buffer shared across requests (process-level singleton)
_api_records: deque = deque(maxlen=RING_BUFFER)


# ─── Model Inference Timing ───────────────────────────────────────────────────

def measure_inference(
    model          : tf.keras.Model,
    img_array      : np.ndarray,
    warmup_runs    : int = 2,
    timed_runs     : int = 5,
) -> dict:
    """
    Measure model inference latency over *timed_runs* forward passes.

    Parameters
    ----------
    model       : loaded Keras model
    img_array   : pre-processed image array shape (1, H, W, 3)
    warmup_runs : number of silent warm-up passes (GPU/cache warm-up)
    timed_runs  : number of timed passes to average

    Returns
    -------
    dict with keys: mean_ms, min_ms, max_ms, std_ms, throughput_rps
    """
    # warm up
    for _ in range(warmup_runs):
        _ = model.predict(img_array, verbose=0)

    latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = model.predict(img_array, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

    mean_ms = statistics.mean(latencies)
    return {
        "mean_ms"        : round(mean_ms, 3),
        "min_ms"         : round(min(latencies), 3),
        "max_ms"         : round(max(latencies), 3),
        "std_ms"         : round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 3),
        "throughput_rps" : round(1000 / mean_ms, 2),
    }


# ─── API Call Recording ───────────────────────────────────────────────────────

def record_api_call(
    endpoint      : str,
    duration_ms   : float,
    status_code   : int,
    model_ms      : Optional[float] = None,
) -> None:
    """
    Push one API timing record into the ring buffer.
    Call this at the end of every API handler.
    """
    _api_records.append({
        "ts"         : datetime.now(timezone.utc).isoformat(),
        "endpoint"   : endpoint,
        "duration_ms": round(duration_ms, 3),
        "status_code": status_code,
        "model_ms"   : round(model_ms, 3) if model_ms is not None else None,
    })


def get_api_stats(last_n: int = 100) -> dict:
    """
    Rolling statistics over the last *last_n* recorded API calls.

    Returns
    -------
    dict with: total_calls, success_rate, mean_ms, p50_ms, p95_ms, p99_ms,
               throughput_rps, recent (last 20 records)
    """
    records = list(_api_records)[-last_n:]
    if not records:
        return {
            "total_calls"  : 0,
            "success_rate" : None,
            "mean_ms"      : None,
            "p50_ms"       : None,
            "p95_ms"       : None,
            "p99_ms"       : None,
            "throughput_rps": None,
            "recent"       : [],
        }

    durations     = [r["duration_ms"] for r in records]
    success_count = sum(1 for r in records if 200 <= r["status_code"] < 300)
    sorted_dur    = sorted(durations)

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return round(data[min(idx, len(data) - 1)], 3)

    # Rough throughput: how many req/s over the window spanned by these records
    throughput = None
    if len(records) >= 2:
        try:
            t0 = datetime.fromisoformat(records[0]["ts"])
            t1 = datetime.fromisoformat(records[-1]["ts"])
            span = (t1 - t0).total_seconds()
            throughput = round(len(records) / span, 2) if span > 0 else None
        except Exception:
            pass

    return {
        "total_calls"   : len(records),
        "success_rate"  : round(success_count / len(records) * 100, 1),
        "mean_ms"       : round(statistics.mean(durations), 3),
        "median_ms"     : percentile(sorted_dur, 50),
        "p95_ms"        : percentile(sorted_dur, 95),
        "p99_ms"        : percentile(sorted_dur, 99),
        "throughput_rps": throughput,
        "recent"        : list(reversed(records[-20:])),
    }


# ─── Model Metrics from Disk ──────────────────────────────────────────────────

def get_model_metrics(model_name: str) -> Optional[dict]:
    """
    Load the metrics JSON saved alongside the model file.
    Returns None if not found.
    """
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    meta_path    = MODELS_DIR / f"{model_name}_meta.json"

    result = {}
    for p in (meta_path, metrics_path):
        if p.exists():
            with open(p) as f:
                result.update(json.load(f))

    return result if result else None





def list_available_models() -> list[dict]:
    """
    Scan models/ and return a list of dicts describing each saved model.
    Supports both .keras and .h5 files.
    """
    models = []

    # combine .keras and .h5 files
    model_files = sorted(
        list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
        key=lambda p: p.stat().st_mtime
    )

    for f in model_files:
        meta = get_model_metrics(f.stem) or {}
        models.append({
            "name"        : f.stem,
            "path"        : str(f),
            "size_mb"     : round(f.stat().st_size / 1e6, 2),
            "timestamp"   : meta.get("timestamp"),
            "val_accuracy": meta.get("final_val_accuracy"),
            "classes"     : meta.get("classes"),
        })

    return models


# ─── Health / Uptime ──────────────────────────────────────────────────────────

def health_check() -> dict:
    uptime_s = time.time() - _START_TIME
    hrs, rem = divmod(int(uptime_s), 3600)
    mins, secs = divmod(rem, 60)
    return {
        "status"        : "ok",
        "uptime_seconds": round(uptime_s, 1),
        "uptime_human"  : f"{hrs}h {mins}m {secs}s",
        "models_dir"    : str(MODELS_DIR),
        "models_count"  : len(list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5"))),
        "api_calls_buffered": len(_api_records),
    }


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(json.dumps(health_check(), indent=2))
    print(json.dumps(list_available_models(), indent=2))
    print(json.dumps(get_api_stats(), indent=2))
