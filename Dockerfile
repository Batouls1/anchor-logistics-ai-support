FROM python:3.11-slim

# System deps: build-essential for compiling packages, ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in model weights at build time -- avoids slow/flaky downloads at runtime
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', compute_type='int8')"

COPY . .

RUN mkdir -p static/audio

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]