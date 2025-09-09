# CPU-first Docker image for reproduction via reproduce_paper.sh
# Note: This image does not contain a GPU/CuPy stack. To regenerate the run NPZ,
# mount a precomputed runs/ folder or use a GPU-enabled base and install cupy-cudaXX.

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repository contents
COPY . /app

# Python deps (minimal, as reproduce_paper.sh will also ensure env)
RUN python -m pip install --upgrade pip && \
    pip install numpy==1.26.* scipy==1.13.* pandas==2.2.* matplotlib==3.8.* dynesty==2.1.*

# Ensure the runner script is executable
RUN chmod +x /app/reproduce_paper.sh

# By default, require host to mount runs/ and external_data/Rotmod_LTG
# Example run (host):
#   docker run --rm -it \
#     -e RUN_DIR=runs/enhanced_20250805_115400 \
#     -e SPARC_DIR=external_data/Rotmod_LTG \
#     -e LENS_CSV=docs/lensing_targets.csv \
#     -v $PWD/runs:/app/runs \
#     -v $PWD/external_data/Rotmod_LTG:/app/external_data/Rotmod_LTG:ro \
#     -v $PWD/results:/app/results \
#     -v $PWD/images:/app/images \
#     dgg-repro

ENTRYPOINT ["/app/reproduce_paper.sh"]

