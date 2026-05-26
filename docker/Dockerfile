FROM python:3.9-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install Python package
# ---------------------------------------------------------------------------
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Post-install asset downloads (baked into the image layer)
# ---------------------------------------------------------------------------
RUN python -m unidic download
RUN python melo/scripts/init_downloads.py

# ---------------------------------------------------------------------------
# Default command: start the Gradio WebUI
# ---------------------------------------------------------------------------
EXPOSE 8888
CMD ["python", "-m", "melo.ui.app", "--host", "0.0.0.0", "--port", "8888"]