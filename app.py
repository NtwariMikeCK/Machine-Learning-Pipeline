"""
app.py  —  Streamlit Frontend
==============================
Full ML Pipeline UI:
  • Model selector (dropdown, polls /models — local + GDrive)
  • PREDICT panel  – upload image → predict → show result + latency
  • RETRAIN panel  – LOCAL mode: upload ZIP → Drive → open Colab
                   – CLOUD mode: upload ZIP → GPU server → background retrain
  • SYSTEM panel   – uptime, model performance vs API performance chart
"""

import io
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from pathlib import Path
from datetime import datetime

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NeuralVision Pipeline",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e8e8f0;
  }
  .stApp { background: #0a0a0f; }

  .metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
  }
  .metric-card:hover { border-color: #6c63ff; }
  .metric-val { font-family: 'Space Mono', monospace; font-size: 2rem; color: #6c63ff; font-weight: 700; }
  .metric-lbl { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

  .pred-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #6c63ff;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    animation: glow 2s ease-in-out infinite alternate;
  }
  @keyframes glow {
    from { box-shadow: 0 0 10px #6c63ff44; }
    to   { box-shadow: 0 0 30px #6c63ff88, 0 0 60px #6c63ff22; }
  }
  .pred-class { font-size: 2.5rem; font-weight: 800; color: #fff; }
  .pred-conf  { font-family: 'Space Mono', monospace; color: #6c63ff; font-size: 1.1rem; margin-top: 6px; }

  .badge-ok    { background:#0d3b1e; color:#4caf50; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
  .badge-warn  { background:#3b2a0d; color:#ff9800; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
  .badge-error { background:#3b0d0d; color:#f44336; padding:3px 10px; border-radius:20px; font-size:0.8rem; }

  .step-box {
    background: #12121f;
    border-left: 3px solid #6c63ff;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
  }
  .step-num { color: #6c63ff; font-family: 'Space Mono', monospace; font-weight: 700; margin-right: 8px; }

  .panel-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6c63ff;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-bottom: 0.3rem;
  }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  [data-testid="stSidebar"] {
    background: #0e0e1a;
    border-right: 1px solid #1e1e3a;
  }
  .stButton>button {
    background: linear-gradient(90deg, #6c63ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s, transform 0.1s;
  }
  .stButton>button:hover { opacity: 0.88; transform: translateY(-1px); }
  .js-plotly-plot .plotly { background: transparent !important; }
  hr { border-color: #1e1e3a; }
</style>
""", unsafe_allow_html=True)


# ─── Config ───────────────────────────────────────────────────────────────────
API_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

COLAB_TRAIN_URL   = st.sidebar.text_input(
    "Colab Train Notebook URL",
    value="https://colab.research.google.com/drive/16ETzbLb4gRsxnR9p8ue5JrtFdGGvDuaV?usp=chrome_ntp#scrollTo=sQE3jINaMDHs",
)
COLAB_RETRAIN_URL = st.sidebar.text_input(
    "Colab Retrain Notebook URL",
    value="https://colab.research.google.com/drive/1EX6d5VYGvrYu-BawMglHCs6nY4-5Pwx4?usp=chrome_ntp#scrollTo=hlT1_-Nk2p13",
)

results = {'0': 'Brain Tumor', '1': 'Healthy'}

st.sidebar.markdown("---")
st.sidebar.markdown("### Deployment Mode")
mode = st.sidebar.radio(
    "Select mode",
    ["🖥️ Local (Colab retraining)", "☁️ Cloud (GPU retraining)"],
    help=("Local: upload data to Google Drive, retrain in Colab.\n"
          "Cloud: retraining runs directly on the server GPU."),
)
IS_CLOUD = mode.startswith("☁️")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=8, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), r.elapsed.total_seconds() * 1000
    except requests.exceptions.HTTPError as e:
        return {"error": str(e), "detail": e.response.text}, 0
    except Exception as e:
        return {"error": str(e)}, 0


def _plotly_layout(title=""):
    return dict(
        title         = title,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#e8e8f0", family="Space Mono"),
        xaxis         = dict(gridcolor="#1e1e3a", linecolor="#2a2a4a"),
        yaxis         = dict(gridcolor="#1e1e3a", linecolor="#2a2a4a"),
        margin        = dict(l=40, r=20, t=40, b=40),
    )


# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title, col_badge = st.columns([1, 6, 2])
with col_logo:
    st.markdown("## ")
with col_title:
    st.markdown("# NeuralVision Pipeline")
    st.markdown('<div class="panel-header">DenseNet Image Classifier · MLOps Dashboard</div>',
                unsafe_allow_html=True)
with col_badge:
    health = _get("/health")
    if "error" not in health:
        gdrive_info = health.get("gdrive", {})
        gdrive_ok   = gdrive_info.get("configured", False)
        gdrive_badge = (
            '<span class="badge-ok">● Drive OK</span>'
            if gdrive_ok else
            '<span class="badge-warn">● Drive ✗</span>'
        )
        st.markdown(
            f'<br><span class="badge-ok">● API ONLINE</span>&nbsp;{gdrive_badge}&nbsp;'
            f'<span style="font-family:Space Mono;font-size:0.7rem;color:#555;">'
            f'uptime {health.get("uptime_human","–")}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<br><span class="badge-error">● API OFFLINE</span>',
                    unsafe_allow_html=True)

st.markdown("---")


# ─── Model Selector ───────────────────────────────────────────────────────────
models_resp  = _get("/models")
local_models  = [m["name"] for m in models_resp.get("local", [])]
gdrive_models = [f"[GDrive] {m['name']}" for m in models_resp.get("gdrive", [])]
all_models    = local_models + gdrive_models

if all_models:
    selected_model = st.selectbox(
        "Active Model",
        ["(auto – latest)"] + all_models,
        help="Select the model to use for predictions. 'auto' uses the newest local model.",
    )
    selected_model = (
        None if selected_model == "(auto – latest)"
        else selected_model.replace("[GDrive] ", "")
    )
else:
    st.warning("⚠️ No trained models found. Train a model first via the Colab notebook.")
    selected_model = None

st.markdown("---")


# ─── View Toggle ──────────────────────────────────────────────────────────────
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Predict"

col_toggle1, col_toggle2 = st.columns(2)
with col_toggle1:
    if st.button(
        "Predict",
        use_container_width=True,
        type="primary" if st.session_state.view_mode == "Predict" else "secondary",
    ):
        st.session_state.view_mode = "Predict"
with col_toggle2:
    if st.button(
        "Retrain",
        use_container_width=True,
        type="primary" if st.session_state.view_mode == "Retrain" else "secondary",
    ):
        st.session_state.view_mode = "Retrain"

view_mode = st.session_state.view_mode
st.markdown("---")

if view_mode == "Predict":
    left_panel  = st.container()
    right_panel = None
else:
    left_panel  = None
    right_panel = st.container()


# ╔══════════════════════════════╗
# ║  PREDICT                     ║
# ╚══════════════════════════════╝
if left_panel:
    with left_panel:
        st.markdown('<div class="panel-header">◈ Predict</div>', unsafe_allow_html=True)
        st.markdown("### Image Prediction")

        uploaded_img = st.file_uploader(
            "Upload an image for classification",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="predict_upload",
        )

        save_img_toggle = st.checkbox("Save image for future retraining", value=True)

        if uploaded_img:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded image", use_column_width=True)

        predict_btn = st.button("⚡ Predict", use_container_width=True)

        if predict_btn:
            if not uploaded_img:
                st.error("Please upload an image first.")
            else:
                with st.spinner("Running inference …"):
                    uploaded_img.seek(0)
                    files = {"file": (uploaded_img.name, uploaded_img, uploaded_img.type)}
                    data  = {}
                    if selected_model:
                        data["model_name"] = selected_model
                    data["save_image"] = str(save_img_toggle).lower()

                    t0 = time.perf_counter()
                    resp, api_ms = _post("/predict", files=files, data=data)
                    wall_ms = (time.perf_counter() - t0) * 1000

                if "error" in resp:
                    st.error(f"Prediction failed: {resp['error']}")
                else:
                    pred_class = resp.get("predicted_class", "Unknown")
                    confidence = resp.get("confidence", 0)
                    inf_ms     = resp.get("inference_time_ms", 0)
                    total_ms   = resp.get("total_latency_ms", wall_ms)

                    st.markdown(f"""
                    <div class="pred-box">
                    <div class="pred-class">🏷️ {results.get(pred_class, pred_class)}</div>
                    <div class="pred-conf">{confidence*100:.1f}% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")

                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{inf_ms:.1f}</div>
                    <div class="metric-lbl">Model ms</div></div>""",
                    unsafe_allow_html=True)
                    m2.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{total_ms:.1f}</div>
                    <div class="metric-lbl">API ms</div></div>""",
                    unsafe_allow_html=True)
                    overhead = total_ms - inf_ms
                    m3.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{overhead:.1f}</div>
                    <div class="metric-lbl">Overhead ms</div></div>""",
                    unsafe_allow_html=True)

                    probs = resp.get("all_probabilities", {})
                    if probs:
                        labels = list(probs.keys())
                        values = [v * 100 for v in probs.values()]
                        colors = ["#6c63ff" if l == pred_class else "#2a2a4a"
                                  for l in labels]
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation="h",
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in values],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            **_plotly_layout("Class Probabilities"),
                            height=max(200, len(labels) * 45),
                            xaxis_title="Probability (%)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        f'<div style="font-family:Space Mono;font-size:0.7rem;color:#555;">'
                        f'model: {resp.get("model_used","–")}</div>',
                        unsafe_allow_html=True,
                    )


# ╔══════════════════════════════╗
# ║  RETRAIN                     ║
# ╚══════════════════════════════╝
if right_panel:
    with right_panel:
        st.markdown('<div class="panel-header">◈ Retrain</div>', unsafe_allow_html=True)
        st.markdown("### Model Retraining")

        # ══════════════════════════════════════════════
        #  CLOUD MODE
        # ══════════════════════════════════════════════
        if IS_CLOUD:
            st.info("☁️ **Cloud mode** – retraining runs on the server GPU automatically.")

            upload_zip = st.file_uploader(
                "Upload new training data (ZIP)",
                type=["zip"],
                key="retrain_upload",
                help="ZIP layout: <class_name>/<image_files>",
            )

            if upload_zip:
                st.success(f"✅ Ready: **{upload_zip.name}** ({upload_zip.size/1e6:.1f} MB)")
                st.markdown("""
                **Expected ZIP structure:**
                ```
                data.zip
                ├── class_a/
                │   └── img001.jpg
                └── class_b/
                    └── img001.jpg
                ```
                """)

            retrain_btn = st.button("🚀 Trigger Retraining", use_container_width=True)

            if retrain_btn:
                if not upload_zip:
                    st.error("Please upload a ZIP file first.")
                else:
                    with st.spinner("Sending data to server …"):
                        upload_zip.seek(0)
                        files = {"file": (upload_zip.name, upload_zip, "application/zip")}
                        resp, _ = _post("/retrain", files=files)

                    if "error" in resp:
                        st.error(f"Retraining failed: {resp['error']}")
                    else:
                        st.success("✅ Retraining job started! Poll status below.")

            st.markdown("#### Retraining Status")
            status = _get("/retrain/status")
            if status:
                running  = status.get("running", False)
                s_status = status.get("status", "idle")
                if running:
                    st.markdown('<span class="badge-warn">⏳ RUNNING</span>',
                                unsafe_allow_html=True)
                    st.progress(0.5, "Training in progress …")
                elif s_status == "success":
                    st.markdown('<span class="badge-ok">✅ COMPLETE</span>',
                                unsafe_allow_html=True)
                    m = status.get("metrics", {})
                    if m:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy",  f"{m.get('accuracy',  0)*100:.1f}%")
                        c2.metric("Precision", f"{m.get('precision', 0)*100:.1f}%")
                        c3.metric("Recall",    f"{m.get('recall',    0)*100:.1f}%")
                    st.caption(
                        f"Model saved: **{status.get('model_name','–')}**  |  "
                        f"Best overall: **{status.get('best_model','–')}**"
                    )
                elif s_status == "error":
                    st.markdown('<span class="badge-error">❌ ERROR</span>',
                                unsafe_allow_html=True)
                    st.error(status.get("error", "Unknown error"))
                else:
                    st.caption("No retraining job has run yet.")

        # ══════════════════════════════════════════════
        #  LOCAL MODE  (Google Drive + Colab)
        # ══════════════════════════════════════════════
        else:
            st.info("🖥️ **Local mode** – no local GPU. New data is sent to Google Drive; "
                    "retraining happens in Google Colab.")

            # ── How it works ──────────────────────────────────────────────────
            st.markdown("#### How it works")
            for num, step in enumerate([
                "Upload your labelled images as a ZIP below.",
                "Click **📤 Upload to Google Drive** — the ZIP is sent to your Drive new-data folder via the API.",
                "Click **Open Retrain Notebook** to open the Colab notebook.",
                "Run all cells in Colab: it loads the new data + old data from Drive, retrains, and saves the model back to Drive.",
                "Return here and click **🔄 Sync Models from Drive** to pull the new model and make it available for predictions.",
            ], 1):
                st.markdown(
                    f'<div class="step-box"><span class="step-num">{num}.</span>{step}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Step 1 & 2: Upload ZIP to Google Drive ────────────────────────
            st.markdown("#### Step 1 — Upload new training data")

            upload_zip_local = st.file_uploader(
                "Select a ZIP of new labelled images",
                type=["zip"],
                key="retrain_local_upload",
                help="ZIP layout: <class_name>/<image_files>",
            )

            if upload_zip_local:
                sz_mb = upload_zip_local.size / 1e6
                st.success(f"✅ **{upload_zip_local.name}** selected ({sz_mb:.1f} MB)")

                col_up, col_dl = st.columns(2)

                # ── Send to Drive via API ──────────────────────────────────────
                with col_up:
                    if st.button("📤 Upload to Google Drive", use_container_width=True):
                        with st.spinner("Uploading to Google Drive …"):
                            upload_zip_local.seek(0)
                            files = {
                                "file": (
                                    upload_zip_local.name,
                                    upload_zip_local,
                                    "application/zip",
                                )
                            }
                            resp, _ = _post("/gdrive/upload-data", files=files)

                        if "error" in resp:
                            st.error(f"Upload failed: {resp.get('error')} — {resp.get('detail','')}")
                        else:
                            st.success(
                                f"✅ Uploaded to Drive!  "
                                f"File ID: `{resp.get('gdrive_id','–')}`"
                            )
                            st.session_state["zip_uploaded_to_drive"] = True

                # ── Fallback: manual download ──────────────────────────────────
                with col_dl:
                    upload_zip_local.seek(0)
                    st.download_button(
                        "💾 Download ZIP (manual upload)",
                        data=upload_zip_local.read(),
                        file_name=upload_zip_local.name,
                        mime="application/zip",
                        use_container_width=True,
                        help="Use this if the auto-upload fails — save the file to your Google Drive manually.",
                    )

            st.markdown("---")

            # ── Step 3: Open Colab ────────────────────────────────────────────
            st.markdown("#### Step 2 — Run Colab retraining notebook")

            col_train, col_retrain = st.columns(2)

            with col_train:
                st.markdown("**First-time training**")
                st.caption("Train the model from scratch on your full dataset.")
                st.markdown(f"[🔗 Open Train Notebook ↗]({COLAB_TRAIN_URL})",
                            unsafe_allow_html=False)

            with col_retrain:
                st.markdown("**Retrain with new data**")
                st.caption("Combines old + new data, retrains, saves model to Drive.")
                st.markdown(f"[🔗 Open Retrain Notebook ↗]({COLAB_RETRAIN_URL})",
                            unsafe_allow_html=False)

            # Highlight if ZIP was just uploaded
            if st.session_state.get("zip_uploaded_to_drive"):
                st.success("✅ Data is on Drive — open the Retrain Notebook and run all cells!")

            st.markdown("---")

            # ── Step 4: Sync models back ──────────────────────────────────────
            st.markdown("#### Step 3 — Pull retrained model from Google Drive")
            st.caption(
                "After the Colab notebook finishes, click below to download the "
                "new model from Drive to the local server and reload it."
            )

            col_sync, col_refresh = st.columns(2)

            with col_sync:
                if st.button("🔄 Sync Models from Drive", use_container_width=True):
                    with st.spinner("Syncing from Google Drive …"):
                        resp, _ = _post("/gdrive/sync-models")

                    if "error" in resp:
                        st.error(f"Sync failed: {resp.get('error')} — {resp.get('detail','')}")
                    else:
                        downloaded = resp.get("downloaded", [])
                        if downloaded:
                            st.success(
                                f"✅ Sync complete! Downloaded: **{', '.join(downloaded)}**"
                            )
                        else:
                            st.info("No new models found on Drive (or all already up-to-date).")
                        st.rerun()   # refresh model dropdown

            with col_refresh:
                if st.button("♻️ Refresh Model List", use_container_width=True):
                    st.rerun()

            # ── Drive status summary ──────────────────────────────────────────
            with st.expander("Google Drive status", expanded=False):
                health_data = _get("/health")
                gdrive_info = health_data.get("gdrive", {})
                if gdrive_info:
                    configured = gdrive_info.get("configured", False)
                    if configured:
                        st.markdown(
                            f'<span class="badge-ok">● Drive configured</span>',
                            unsafe_allow_html=True,
                        )
                        st.write(
                            f"**Models folder ID:** `{gdrive_info.get('models_folder_id','–')}`"
                        )
                        st.write(
                            f"**New-data folder ID:** `{gdrive_info.get('newdata_folder_id','–')}`"
                        )
                        remote_models = gdrive_info.get("remote_models", [])
                        if remote_models:
                            st.markdown("**Models on Drive:**")
                            for m in remote_models:
                                st.markdown(
                                    f"- `{m['name']}` — {m['size_mb']} MB "
                                    f"(modified: {m.get('modifiedTime','–')[:10]})"
                                )
                        else:
                            st.caption("No models found on Drive yet.")
                    else:
                        st.markdown(
                            '<span class="badge-error">● Drive not configured</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("""
                        **To configure Google Drive:**
                        1. Enable the Drive API in [Google Cloud Console](https://console.cloud.google.com/).
                        2. Create a Service Account and download the JSON key.
                        3. Place the key at `credentials/gdrive_service_account.json`.
                        4. Set environment variables:
                           - `GDRIVE_MODELS_FOLDER_ID`
                           - `GDRIVE_NEWDATA_FOLDER_ID`
                        5. Share both Drive folders with the service-account email.
                        6. Restart the API server.
                        """)
                else:
                    st.warning("Could not retrieve Drive status from the API.")


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PERFORMANCE SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## System Performance")

api_stats = _get("/system/stats")

if "error" not in api_stats and api_stats.get("total_calls", 0) > 0:
    s1, s2, s3, s4, s5 = st.columns(5)
    kpis = [
        (api_stats.get("total_calls",     "–"),           "Total Calls"),
        (f'{api_stats.get("success_rate","–")}%',         "Success Rate"),
        (f'{api_stats.get("mean_ms",      "–")} ms',      "Avg Latency"),
        (f'{api_stats.get("median_ms",    "–")} ms',      "P50 Latency"),
        (f'{api_stats.get("p95_ms",       "–")} ms',      "P95 Latency"),
    ]
    for col, (val, lbl) in zip([s1, s2, s3, s4, s5], kpis):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div></div>""",
          unsafe_allow_html=True)

    st.markdown("")

    recent = api_stats.get("recent", [])
    if recent:
        df = pd.DataFrame(recent)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

        fig_timeline = go.Figure()
        for ep in df["endpoint"].unique():
            mask = df["endpoint"] == ep
            fig_timeline.add_trace(go.Scatter(
                x    = df[mask]["ts"],
                y    = df[mask]["duration_ms"],
                mode = "lines+markers",
                name = ep,
                line = dict(width=2),
            ))

        model_df = df[df["model_ms"].notna()]
        if not model_df.empty:
            fig_timeline.add_trace(go.Scatter(
                x     = model_df["ts"],
                y     = model_df["model_ms"],
                mode  = "markers",
                name  = "Model inference",
                marker= dict(symbol="diamond", size=9, color="#ffd700"),
            ))

        fig_timeline.update_layout(
            **_plotly_layout("Request Latency Timeline"),
            height=300,
            yaxis_title="Latency (ms)",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        ep_avg = df.groupby("endpoint")["duration_ms"].mean().reset_index()
        ep_avg.columns = ["endpoint", "avg_ms"]
        fig_ep = go.Figure(go.Bar(
            x    = ep_avg["endpoint"],
            y    = ep_avg["avg_ms"],
            marker_color = "#6c63ff",
            text = [f"{v:.1f} ms" for v in ep_avg["avg_ms"]],
            textposition = "outside",
        ))
        fig_ep.update_layout(**_plotly_layout("Avg Latency by Endpoint"),
                             height=280, yaxis_title="ms")
        st.plotly_chart(fig_ep, use_container_width=True)

else:
    st.info("No API calls recorded yet. Make a prediction to see live stats.")
    endpoints = ["/predict", "/retrain", "/gdrive/upload-data", "/gdrive/sync-models", "/models"]
    fake_ms   = [85, 15000, 3000, 5000, 12]
    fig_fake = go.Figure(go.Bar(
        x=endpoints, y=fake_ms,
        marker_color="#2a2a4a",
        text=[f"{v} ms" for v in fake_ms],
        textposition="outside",
    ))
    fig_fake.update_layout(**_plotly_layout("Expected Latency by Endpoint (demo)"),
                           height=280, yaxis_title="ms")
    st.plotly_chart(fig_fake, use_container_width=True)