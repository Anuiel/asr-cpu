#!/usr/bin/env bash
#
# Full benchmark pipeline: setup, data download, run all benchmarks.
# Intended for a fresh Linux machine with Python 3.10+.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
DATA_DIR="data/librispeech"
RESULTS_DIR="results"
REPORTS_DIR="reports"
MODELS="small medium"
BACKENDS="openai-whisper faster-whisper torch-opt"
MAX_SAMPLES=30
SHORT_MAX_DUR=10
LONG_MIN_DUR=30

# ---------- helpers ----------

log() { echo "=== $(date '+%H:%M:%S') $*" >&2; }

run_bench() {
    local backend="$1" model="$2" label="$3"
    shift 3
    local outfile="${RESULTS_DIR}/${label}_${backend}_${model}.json"

    log "Running: ${backend} ${model} ${label}"
    python run_benchmark.py \
        --backend "$backend" \
        --dataset librispeech \
        --data-dir "$DATA_DIR" \
        "$@" \
        --max-samples "$MAX_SAMPLES" \
        --model-name "$model" \
        --output "$outfile"
    log "Saved: $outfile"
}

# ---------- 1. Create venv & install ----------

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment"
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

log "Installing dependencies"
pip install --upgrade pip
pip install -e ".[all]"

# ---------- 2. Download data ----------

if [ ! -f "${DATA_DIR}/manifest.json" ]; then
    log "Downloading LibriSpeech test-clean"
    pip install torchcodec
    python prepare_data.py --output "$DATA_DIR" --split test.clean
else
    log "Data already prepared at ${DATA_DIR}"
fi

# ---------- 3. Run benchmarks ----------

mkdir -p "$RESULTS_DIR" "$REPORTS_DIR"

for backend in $BACKENDS; do
    for model in $MODELS; do
        run_bench "$backend" "$model" "short" --max-duration "$SHORT_MAX_DUR"
        run_bench "$backend" "$model" "long"  --min-duration "$LONG_MIN_DUR"
    done
done

# ---------- 4. Generate summary report ----------

log "Generating summary report"
python - <<'PYSCRIPT'
import json, os, datetime
from benchmark.runner import collect_hardware_info
from benchmark.metrics import corpus_wer, corpus_cer

results_dir = "results"
rows = []

for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith(".json") or fname.startswith("test"):
        continue
    with open(os.path.join(results_dir, fname)) as f:
        data = json.load(f)
    samples = data.get("samples", data)
    if not samples:
        continue

    parts = fname.replace(".json", "").split("_")
    duration_bucket = parts[0]
    backend = parts[1]
    model = parts[2] if len(parts) > 2 else "?"

    n = len(samples)
    total_audio = sum(s["duration_sec"] for s in samples)
    avg_rtf = sum(s["rtf"] for s in samples) / n
    total_wall = sum(s["wall_time_sec"] for s in samples)
    max_rss = max(s["peak_rss_mb"] for s in samples)
    avg_energy = sum(s["energy_estimate_j"] for s in samples) / n
    refs = [s["reference"] for s in samples]
    hyps = [s["hypothesis"] for s in samples]
    wer = corpus_wer(refs, hyps) * 100
    cer = corpus_cer(refs, hyps) * 100

    rows.append({
        "duration": duration_bucket, "backend": backend, "model": model,
        "samples": n, "total_audio_s": round(total_audio, 1),
        "avg_rtf": round(avg_rtf, 3), "total_wall_s": round(total_wall, 1),
        "peak_rss_mb": round(max_rss), "avg_energy_j": round(avg_energy, 2),
        "wer_pct": round(wer, 2), "cer_pct": round(cer, 2),
    })

hw = collect_hardware_info()
report = {"hardware": hw, "benchmarks": rows}
outpath = f"reports/benchmark_summary.json"
with open(outpath, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report saved to {outpath}")
PYSCRIPT

log "Done! Results in ${RESULTS_DIR}/, report in ${REPORTS_DIR}/"
