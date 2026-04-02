FROM python:3.10-slim

WORKDIR /app

# System deps
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: start the FastAPI server
ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "src.prediction:app", "--host", "0.0.0.0", "--port", "8000"]
