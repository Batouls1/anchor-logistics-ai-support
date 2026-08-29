FROM python:3.11-slim

# build-essential to compile packages, ffmpeg to decode audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# --index-url, not --extra-index-url: an extra index still lets pip
# resolve torch from PyPI, which is the multi-gigabyte CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 --index-url https://pypi.org/simple -r requirements.txt

# Bake model weights in, outside /root so the non-root user below can
# still read them -- otherwise every model re-downloads on first start.
ENV HF_HOME=/opt/model-cache
RUN mkdir -p /opt/model-cache

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', compute_type='int8')"

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
