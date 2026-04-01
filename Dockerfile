FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY benchmark/ benchmark/
COPY run_benchmark.py .

RUN pip install --no-cache-dir -e ".[all]"

ENTRYPOINT ["python", "run_benchmark.py"]
