# builder: compiles and downloads, then is discarded 
FROM python:3.11-slim AS builder

# The C toolchain exists only in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --index-url, not --extra-index-url: an extra index still lets pip
# resolve torch from PyPI, which is the multi-gigabyte CUDA build.
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

# Baked here so the runtime stage copies one cache directory wholesale.
# Models in the image mean no cold download mid-conversation, and the
# container runs offline.
ENV HF_HOME=/opt/model-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', compute_type='int8')"


# ---------- runtime ----------
FROM python:3.11-slim

# ffmpeg decodes voice-note audio. libgomp1 is the OpenMP runtime torch
# and ctranslate2 link against: it arrives free with build-essential in
# the builder, so it must be requested explicitly here or importing torch
# fails on a missing libgomp.so.1.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Installed packages and console scripts, without the toolchain that built them.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/model-cache /opt/model-cache

# PYTHONUNBUFFERED matters in a container: without it Python buffers
# stdout when it isn't a terminal, so `docker logs` lags or looks empty
# exactly when something has gone wrong.
ENV HF_HOME=/opt/model-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . .

RUN mkdir -p rag/bm25_store

# Windows editors save shell scripts with CRLF, which fails here as a
# confusing "no such file or directory" on a script that clearly exists.
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

# Docker seeds a fresh named volume from the image path including
# ownership, so this chown is what lets the entrypoint write chunks.pkl.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/model-cache
USER appuser

EXPOSE 8000

# Long start-period: a fresh volume runs the RAG ingest (several minutes)
# before uvicorn binds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=360s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]