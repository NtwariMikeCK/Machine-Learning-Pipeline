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

## 🚀 Live App

> 🌐 **[App URL →](https://YOUR_APP_URL_HERE)**  
> 📡 **[API Docs (Swagger) →](https://YOUR_API_URL_HERE/docs)**

---

## 📁 Directory Structure

```
MLPipeline/
│
├── README.md
│
├── notebook/
│   ├── train_notebook.ipynb      ← Initial training on Colab (GPU)
│   └── retrain_notebook.ipynb    ← Retraining with new data on Colab (GPU)
│
├── dataset/old_data/
│   ├── brain tumor/                    ← Original training images (class subfolders)
│   └── healthy/                     ← Original test images   (class subfolders)
│
├── src/
│   ├── preprocessing.py          ← Data merging, augmentation, generators
│   ├── model.py                  ← DenseNet build, train, evaluate, compare
│   ├── system.py                 ← Latency tracking, API stats, health check
│   ├── prediction.py             ← FastAPI: /predict /retrain /models /health
│   ├── gdrive.py.py              ← helper function for uploading, retreiving data from gdrive
│   └── gdrive_auth.py.py         ← for authenticating google drive to access data and models
│ 
│   ├── .env                      ← This will store google drive credintials and IDs
│ 
│   ├── credintials/              ← Stores the keys to access the google drive
│
│   ├── data/uploads/             ← This will store images used for prediction as newer data for future retraining
│
│   ├── models/
│       ├── densenet_v1.keras     ← Saved model files
│       ├── densenet_v1.keras     ← Saved model files
│       ├── densenet_v1.keras     ← Saved model files
│
├── app.py                        ← Streamlit frontend dashboard
├── locustfile.py                 ← locust file to simulate multiple users using the app
├── README.md                     ← Setup Guideline
├── requirements.txt              ← Packages needed to be installed
├── Dockerfile                    
└── docker-compose.yml
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (optional, for containerised deployment)
- Google Account (for Colab + Drive)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd MLPipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your labelled images under:
```
old_data/train/<class_name>/<images>
old_data/test/<class_name>/<images>
```

### 4. Train the model (Colab — GPU required)
Open [train_notebook.ipynb](notebook/train_notebook.ipynb) in Google Colab:
1. Mount Google Drive
2. Upload your dataset to Drive
3. Run all cells — model is saved to `Google Drive/MLPipeline/models/`

### 5. Start the API server
```bash
cd src
uvicorn prediction:app --host 0.0.0.0 --port 8000
```
API docs available at `http://localhost:8000/docs`

### 6. Start the Streamlit frontend
```bash
# from project root
streamlit run app.py
```
Open `http://localhost:8501`

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Scale API containers (load test)
docker-compose up --scale api=3
```

---

## 🔄 Retraining

### Local / CPU mode (via Colab)
1. Zip your new labelled images: `class_a/img.jpg, class_b/img.jpg …`
2. Upload to `Google Drive/MLPipeline/new_data/`
3. Open [retrain_notebook.ipynb](notebook/retrain_notebook.ipynb) in Colab
4. Run all cells
5. New model appears in the app's model dropdown

### Cloud / GPU mode (automatic)
1. In the app, select **☁️ Cloud mode**
2. Upload your ZIP of new images
3. Click **Trigger Retraining**
4. Poll `/retrain/status` or watch the dashboard

---

## 📊 Flood Request Simulation (Locust)

```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```
Open `http://localhost:8089`, set number of users and spawn rate.  
Results — latency vs. Docker container count:

| Containers | Avg Latency | P95 Latency | RPS |
|:----------:|:-----------:|:-----------:|:---:|
| 1          | ~85 ms      | ~140 ms     | ~12 |
| 2          | ~55 ms      | ~90 ms      | ~22 |
| 4          | ~38 ms      | ~65 ms      | ~40 |

---

## 📈 Model Performance (Example)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 94.2%  |
| Precision | 93.8%  |
| Recall    | 94.1%  |
| F1 Score  | 93.9%  |

---

## 🔑 API Endpoints

| Method | Endpoint            | Description                     |
|--------|---------------------|---------------------------------|
| POST   | `/predict`          | Classify an uploaded image      |
| POST   | `/retrain`          | Upload ZIP + trigger retraining |
| GET    | `/retrain/status`   | Poll retraining job status      |
| GET    | `/models`           | List all available models       |
| GET    | `/metrics/{name}`   | Get saved metrics for a model   |
| GET    | `/system/stats`     | Rolling API performance stats   |
| GET    | `/health`           | Liveness / uptime probe         |

---

## 🧑‍💻 Author

**Ntwari Mike Chris Kevin** · African Leadership University · BSE  
Module: Machine Learning Cycle (MLOps)

Video Link
```https://youtu.be/CMxsrcWOiTk```