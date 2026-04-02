"""
locustfile.py  —  Flood request simulation
-------------------------------------------
Usage:
    locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to set number of users and spawn rate.

Test scenarios:
  • PredictUser   – sends POST /predict with a random image
  • StatsUser     – polls GET /system/stats and /health
  • ModelsUser    – polls GET /models
"""

import io
import random
from PIL import Image
from locust import HttpUser, task, between, events


def _make_fake_image(size=(224, 224)) -> bytes:
    """Generate a random RGB image as JPEG bytes."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    img = Image.new("RGB", size, color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class PredictUser(HttpUser):
    """Simulates users uploading images for prediction."""
    wait_time = between(0.5, 2.0)   # seconds between tasks
    weight    = 70                   # 70% of virtual users

    @task(10)
    def predict_image(self):
        img_bytes = _make_fake_image()
        self.client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            name="/predict",
        )

    @task(2)
    def check_health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def list_models(self):
        self.client.get("/models", name="/models")


class StatsUser(HttpUser):
    """Simulates a monitoring dashboard polling stats."""
    wait_time = between(1.0, 5.0)
    weight    = 20                   # 20% of virtual users

    @task(5)
    def get_stats(self):
        self.client.get("/system/stats", name="/system/stats")

    @task(3)
    def health(self):
        self.client.get("/health", name="/health")

    @task(2)
    def models(self):
        self.client.get("/models", name="/models")


class BurstUser(HttpUser):
    """Simulates burst traffic — very fast repeated predictions."""
    wait_time = between(0.1, 0.5)
    weight    = 10                   # 10% of virtual users

    @task
    def burst_predict(self):
        img_bytes = _make_fake_image(size=(224, 224))
        self.client.post(
            "/predict",
            files={"file": ("burst.jpg", img_bytes, "image/jpeg")},
            name="/predict [burst]",
        )


# ─── Event hooks for custom reporting ────────────────────────────────────────

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Log slow requests (> 500 ms) to console."""
    if response_time > 500:
        print(f"[SLOW] {request_type} {name} took {response_time:.0f} ms")
