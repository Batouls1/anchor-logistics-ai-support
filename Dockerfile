FROM python:3.11-slim

# System deps: build-essential for compiling packages, ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 1. Install CPU PyTorch using extra-index-url
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# 2. COPY requirements.txt and install remaining packages with high network tolerance
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 --index-url https://pypi.org/simple -r requirements.txt

# Bake in model weights at build time
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', compute_type='int8')"

COPY . .

RUN mkdir -p static/audio

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]