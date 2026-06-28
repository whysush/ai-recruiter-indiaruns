# Sandbox / demo image (submission_spec.md §10.5 — "self-contained docker run
# recipe ... must build and run unmodified").
#
# This image runs the REAL timed ranking step on a small bundled sample
# (data/sample_candidates.json, 50 candidates) end-to-end and writes a ranked CSV.
# It needs ONLY numpy: the committed embedding artifacts in artifacts/ supply the
# semantic component, so the container runs fully offline, CPU-only, in seconds —
# exactly mirroring the constraints in §3.
#
# Build:  docker build -t airecruiter .
# Run:    docker run --rm airecruiter
#         (writes /app/sample_submission_out.csv and prints the top rows)
# Run on your own ≤100-candidate file:
#         docker run --rm -v "$PWD/mysample.json:/app/mysample.json" airecruiter \
#                python scripts/run_submission.py --candidates mysample.json \
#                --cache artifacts/ --out out.csv

FROM python:3.11-slim

WORKDIR /app

# Ranking step depends only on numpy (no torch, no network at runtime).
RUN pip install --no-cache-dir "numpy>=1.24,<3.0"

# Copy the code, the committed embedding artifacts, the JD, and the sample.
COPY airecruiter/ ./airecruiter/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY data/job_description.txt ./data/job_description.txt
COPY data/sample_candidates.json ./data/sample_candidates.json

# Default: rank the bundled sample end-to-end using the committed embeddings.
CMD ["sh", "-c", "python scripts/run_submission.py --candidates data/sample_candidates.json --cache artifacts/ --out sample_submission_out.csv --top-k 25 && echo '--- sample_submission_out.csv (top 6) ---' && head -7 sample_submission_out.csv"]
