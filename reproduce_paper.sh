#!/usr/bin/env bash
set -euo pipefail

# Reproduce paper figures/tables end-to-end.
# - Creates a local venv, installs minimal deps + in-repo utils
# - Optionally regenerates the dynesty run (GPU/CuPy required) if RUN_GENERATE=1
# - Runs the orchestrator via scripts/reproduce_paper.py (paper preset)
# - Adds MW Kz plot with overlay bands if the CSV exists
#
# Config via env vars (override as needed):
RUN_DIR="${RUN_DIR:-runs/rar_plateau_mw_full}"
SPARC_DIR="${SPARC_DIR:-external_data/Rotmod_LTG}"
LENS_CSV="${LENS_CSV:-docs/lensing_targets.csv}"
PRESET="${PRESET:-paper}"
SAMPLE="${SAMPLE:-q2plus}"
HIER_A0="${HIER_A0:-0}"
HIER_A0_BAYES="${HIER_A0_BAYES:-0}"
RUN_GENERATE="${RUN_GENERATE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${THREADS:-8}"
# Runner knobs (only if RUN_GENERATE=1 and GPU/CuPy available)
NLIVE="${NLIVE:-2000}"
MAXCALL="${MAXCALL:-1500000}"
DLOGZ="${DLOGZ:-0.01}"
SEED="${SEED:-42}"

say() { printf "\n[repro] %s\n" "$*"; }

# 0) LFS (best effort)
say "Ensuring Git LFS objects are available (best effort)"
if command -v git >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
  git lfs pull || true
fi

# 1) Python env
say "Setting up Python environment"
if [ ! -d ".venv" ]; then
  "${PYTHON_BIN}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
# Base deps: use in-repo utils and dynesty per WARP.md
python -m pip install ./utils >/dev/null
python -m pip install numpy scipy pandas matplotlib dynesty >/dev/null

# 2) Optionally regenerate run (GPU/CuPy strongly recommended)
find_npz() {
  python - "$RUN_DIR" << 'PY'
import sys, pathlib, numpy as np
run_dir = pathlib.Path(sys.argv[1])
if not run_dir.exists():
    print("")
    sys.exit(0)
# prefer NPZ with samples/param_names
cands = sorted(run_dir.glob("*.npz"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
for p in cands:
    try:
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            if {'samples','param_names'} <= files or {'best_params','param_names'} <= files:
                print(str(p))
                sys.exit(0)
    except Exception:
        pass
print(str(cands[0]) if cands else "")
PY
}

npz_path="$(find_npz)"

if [ "${RUN_GENERATE}" = "1" ]; then
  say "RUN_GENERATE=1: attempting to regenerate dynesty run in ${RUN_DIR}"
  if python -c 'import cupy' >/dev/null 2>&1; then
    if [ -f "runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py" ]; then
      RUNNER="runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py"
    else
      RUNNER="runners/run_dynesty_stellar_fit_cupy.py"
    fi
    python "$RUNNER" \
      --xi rar_plateau \
      --nlive "$NLIVE" --maxcall "$MAXCALL" --dlogz_target "$DLOGZ" \
      --seed "$SEED" --num_threads "$THREADS" \
      --run_analysis \
      --out "$RUN_DIR"
    npz_path="$(find_npz)"
  else
    say "CuPy not found; cannot regenerate run on CPU-only. Provide a GPU/CuPy env or pre-populate $RUN_DIR with an NPZ."
  fi
fi

# Require a valid NPZ to avoid silent fallback that would diverge from paper
if [ -z "$npz_path" ] || [ ! -f "$npz_path" ]; then
  say "ERROR: No run NPZ found under $RUN_DIR."
  say "Provide a paper run directory (with NPZ) via RUN_DIR, OR set RUN_GENERATE=1 on a GPU/CuPy machine."
  exit 2
fi
say "Using NPZ: $npz_path"

# 3) Reproduce main paper artifacts via Python wrapper
cmd=(
  python scripts/reproduce_paper.py
  --run-dir "$RUN_DIR"
  --sparc-dir "$SPARC_DIR"
  --lensing-csv "$LENS_CSV"
  --preset "$PRESET"
  --sample "$SAMPLE"
)
if [ "$HIER_A0" = "1" ]; then cmd+=( --hierarchical-a0 ); fi
if [ "$HIER_A0_BAYES" = "1" ]; then cmd+=( --hierarchical-a0-bayes ); fi
say "Running: ${cmd[*]}"
"${cmd[@]}"

# 4) Milky Way Kz (full 3D) with overlay bands and optional baryon prior propagation
if [ -f "docs/mw_kz_overlay_two_bands.csv" ]; then
  say "Adding MW Kz with overlay bands (and prior band if enabled)"
  python scripts/next_steps_from_run.py \
    --preset "$PRESET" \
    --run-dir "$RUN_DIR" \
    --sparc-dir "$SPARC_DIR" \
    --mw-kz \
    --mw-kz-overlay-csv docs/mw_kz_overlay_two_bands.csv \
    ${MW_KZ_PRIOR_BAND:+--mw-kz-prior-band} \
    ${MW_KZ_PRIOR_BAND:+--mw-prior-samples ${MW_PRIOR_SAMPLES:-128}}
else
  say "MW Kz overlay CSV not found (docs/mw_kz_overlay_two_bands.csv); skipping overlay bands"
fi

say "Done. Artifacts:"
say "  results/next_steps/$(basename "$RUN_DIR")/"
say "  images/next_steps/$(basename "$RUN_DIR")/"

