# 🧠 NeuralVision Pipeline
### DenseNet Image Classifier · MLOps Assignment

---

## 📌 Project Description

An end-to-end ML pipeline built on **DenseNet121** for image classification.  
The system supports prediction, model retraining (both locally via Colab and on the cloud with GPU), live performance monitoring, and a full-featured Streamlit dashboard.

---

## 🎬 Video Demo

> 📺 **[YouTube Demo Link →](https://youtu.be/CMxsrcWOiTk)**

---

## 📁 Directory Structure

```
MLPipeline/
│
├── README.md
│
├── notebook/
│   ├── train_notebook.ipynb          ← Initial training on Colab (GPU)
│   └── retrain_notebook.ipynb        ← Retraining with new data on Colab (GPU)
│
├── dataset/old_data/
│   ├── brain tumor/                  ← Original training images (class subfolders)
│   └── healthy/                      ← Original test images   (class subfolders)
│
├── src/
│   ├── preprocessing.py              ← Data merging, augmentation, generators
│   ├── model.py                      ← DenseNet build, train, evaluate, compare
│   ├── system.py                     ← Latency tracking, API stats, health check
│   ├── prediction.py                 ← FastAPI: /predict /retrain /models /health
│   ├── gdrive.py                     ← Helper functions for Google Drive access
│   ├── gdrive_auth.py                ← One-time OAuth2 authentication script
│   ├── .env                          ← Your Google Drive folder IDs (never commit)
│   ├── credentials/
│   │   ├── oauth_client_secret.json  ← Downloaded from Google Cloud Console
│   │   └── token.json                ← Auto-generated after running gdrive_auth.py
│   ├── data/uploads/                 ← Saved prediction images for future retraining
│   └── models/
│       ├── densenet_best.keras       ← Saved model files (pulled from Drive on startup)
│       └── ...
│
├── app.py                            ← Streamlit frontend dashboard
├── locustfile.py                     ← Locust load testing
├── requirements.txt                  ← Python dependencies
├── Dockerfile
└── docker-compose.yml
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (optional, for containerised deployment)
- A Google Account (for Google Drive + Colab)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/NtwariMikeCK/Machine-Learning-Pipeline.git
cd MLPipeline
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Google Drive setup

The pipeline uses Google Drive to store trained models and new training data.
Every user who wants to run the full pipeline (including retraining) needs to
connect their own Google Drive. Follow these steps exactly.

#### 3a — Enable the Google Drive API

1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create a new project (or select an existing one).
3. Navigate to **APIs & Services → Library**.
4. Search for **"Google Drive API"** and click **Enable**.

#### 3b — Create OAuth2 credentials

> ⚠️ Use **OAuth 2.0 Client ID**, NOT a Service Account.
> Service accounts cannot upload to personal Google Drive storage.

1. Go to **APIs & Services → Credentials**.
2. Click **+ Create Credentials → OAuth client ID**.
3. If prompted to configure the consent screen:
   - Choose **External** → Fill in app name (anything) → Save.
   - Under **Scopes**, add `.../auth/drive` → Save.
   - Under **Test users**, add your own Google email → Save.
4. Back on the Credentials page, click **+ Create Credentials → OAuth client ID** again.
5. Application type: **Desktop app** → Name it anything → **Create**.
6. Click **Download JSON** on the confirmation dialog.
7. Save the downloaded file as:
   ```
   src/credentials/oauth_client_secret.json
   ```

#### 3c — Create Google Drive folders

In your Google Drive, create the following two folders:

```
My Drive/
└── MLPipeline/
    ├── models/       ← trained model files will be saved here
    ├── old_data/       ← old data used in training
    └── new_data/     ← new training ZIPs uploaded from the app go here
```

To get each folder's ID:
- Open the folder in Google Drive in your browser.
- Copy the last segment of the URL:
  ```
  https://drive.google.com/drive/folders/1ABCxyz123...
                                          ↑ this is the folder ID
  ```

#### 3d — Create your `.env` file

Inside the `src/` folder, create a file called `.env`:

```bash
cd src
cp .env.example .env   # if .env.example exists, otherwise create it manually
```

Fill in your folder IDs:

```dotenv
GDRIVE_MODELS_FOLDER_ID=your_models_folder_id_here
GDRIVE_NEWDATA_FOLDER_ID=your_newdata_folder_id_here
GDRIVE_CREDENTIALS_PATH=credentials/oauth_client_secret.json
```

> ⚠️ No quotes, no spaces around `=`, no `?usp=sharing` at the end of IDs.

#### 3e — Run the one-time authentication script

This opens a browser window for you to log in with your Google account.
It only needs to be run **once** — it saves a `token.json` that the server
reuses automatically (and refreshes when it expires).

```bash
cd src
python gdrive_auth.py
```

A browser window will open → log in with your Google account → click **Allow**.

You should see:
```
✅ Token saved to credentials/token.json
```

> If the browser does not open automatically, copy the URL printed in the
> terminal and paste it into your browser manually.

---

### Step 4 — Add your dataset
Download the old data from ```https://drive.google.com/drive/folders/1c_IP4M4asGd1o3CLZE9YZQAcBfM4FkFr?usp=drive_link```
place the data in your google drive old_data
Place your labelled images under:

```
src/old_data/<class_name>/<images>
```

Example for brain tumour detection:
```
src/old_data/brain tumor/img001.jpg
src/old_data/healthy/img001.jpg
src/old_data/brain tumor/img001.jpg
src/old_data/healthy/img001.jpg
```

---

### Step 5 — Train the model (Colab — GPU required)

Open [train_notebook.ipynb](notebook/train_notebook.ipynb) in Google Colab:

1. Mount your Google Drive when prompted.
2. Upload your dataset to Drive under `MLPipeline/old_data/`.
3. Run all cells — the trained model is saved to `My Drive/MLPipeline/models/`.

---

### Step 6 — Start the API server

```bash
cd src
uvicorn prediction:app --host 0.0.0.0 --port 8000 --reload
```

On startup the server will automatically pull the latest models from your
Google Drive into `src/models/`. You should see in the terminal:

```
[STARTUP] Syncing models from Google Drive …
[GDrive] Downloaded → models/densenet_best.h5
[STARTUP] Startup complete.
```

API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Step 7 — Start the Streamlit frontend

In a **separate terminal** from the project root:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🔄 Retraining

### 🖥️ Local / CPU mode (via Colab)

Use this mode when running on a laptop without a GPU.

1. In the app, select **🖥️ Local (Colab retraining)** in the sidebar.
2. Prepare a ZIP of your new labelled images with this structure:
   ```
   new_data.zip
   ├── brain tumor/
   │   └── img001.jpg
   └── healthy/
       └── img001.jpg
   ```
3. Upload the ZIP in the **Retrain** panel — click **📤 Upload to Google Drive**.
4. Click the **Open Retrain Notebook** link to open Colab.
5. Run all cells in Colab — it loads the new data from Drive, combines it with
   the old dataset, retrains the model, and saves it back to Drive.
6. Return to the app and click **🔄 Sync Models from Drive** to pull the new
   model and make it available for predictions immediately.

### ☁️ Cloud / GPU mode (automatic)

Use this mode when the API is deployed on a cloud machine with a GPU.

1. In the app, select **☁️ Cloud (GPU retraining)** in the sidebar.
2. Upload your ZIP of new labelled images in the **Retrain** panel.
3. Click **🚀 Trigger Retraining** — training runs in the background on the server.
4. Watch the status badge or poll `/retrain/status` for progress.

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Scale API containers (load test)
docker-compose up --scale api=3
```

> When running in Docker, make sure your `.env` file and `credentials/` folder
> are inside `src/` before building — they are copied into the container by
> the Dockerfile.

---

## 📊 Load Testing (Locust)

```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

Open [http://localhost:8089](http://localhost:8089), set the number of users and spawn rate.

Results — latency vs. Docker container count:

| Containers | Avg Latency | P95 Latency | RPS |
|:----------:|:-----------:|:-----------:|:---:|
| 1          | ~85 ms      | ~140 ms     | ~12 |
| 2          | ~55 ms      | ~90 ms      | ~22 |
| 4          | ~38 ms      | ~65 ms      | ~40 |

---

## 🔑 API Endpoints

| Method | Endpoint                  | Description                                        |
|--------|---------------------------|----------------------------------------------------|
| POST   | `/predict`                | Classify an uploaded image                         |
| POST   | `/retrain`                | Upload ZIP + trigger retraining (cloud/GPU mode)   |
| GET    | `/retrain/status`         | Poll retraining job status                         |
| GET    | `/models`                 | List all available models (local + Google Drive)   |
| GET    | `/metrics/{name}`         | Get saved evaluation metrics for a model           |
| GET    | `/system/stats`           | Rolling API performance statistics                 |
| GET    | `/health`                 | Liveness probe + Google Drive connection status    |
| POST   | `/gdrive/upload-data`     | Upload new-data ZIP to Google Drive (local mode)   |
| POST   | `/gdrive/sync-models`     | Pull latest models from Google Drive               |

---

## 🔒 Security & `.gitignore`

Make sure the following are in your `.gitignore` so credentials are never
committed to GitHub:

```gitignore
# Google Drive credentials
src/credentials/
src/.env


# Model files (large)
src/models/*.keras

# Data
src/data/
src/old_data/
```

---

## 🧑‍💻 Author

**Ntwari Mike Chris Kevin** · African Leadership University  
Module: Machine Learning Cycle (MLOps)
