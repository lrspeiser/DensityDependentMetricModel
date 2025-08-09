#!/usr/bin/env bash
# fetch_sparc.sh
# Purpose: Download SPARC master sheet + per-galaxy HIrad/SB files into your expected directory
# Usage:   ./scripts/fetch_sparc.sh external_data/Rotmod_LTG
# Notes:
#  - Handles the leading-zero directory names used by SPARC (e.g., NGC0598 for M33, UGC0128).
#  - Converts the SPARC machine-readable table (.mrt) to CSV for your loader.
#  - Emits clear console logs at each step.

set -euo pipefail

DEST_DIR="${1:-external_data/Rotmod_LTG}"
BASE_URL="https://astroweb.cwru.edu/SPARC"
DATA_URL="${BASE_URL}/data"
MASTER_MRT_URL="${BASE_URL}/SPARC_Lelli2016c.mrt"

echo ">>> [SPARC] Target directory: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

###############################################################################
# 1) Download the master sheet (machine-readable table) and convert to CSV
###############################################################################
echo ">>> [SPARC] Fetching master sheet (.mrt) ..."
MASTER_MRT_PATH="${DEST_DIR}/MasterSheet_SPARC.mrt"
MASTER_CSV_PATH="${DEST_DIR}/MasterSheet_SPARC.csv"

curl -fL --retry 3 --retry-delay 2 -o "${MASTER_MRT_PATH}" "${MASTER_MRT_URL}"
echo ">>> [SPARC] Saved: ${MASTER_MRT_PATH}"

echo ">>> [SPARC] Converting master sheet .mrt -> .csv (for your loader) ..."
MASTER_MRT_PATH="${MASTER_MRT_PATH}" MASTER_CSV_PATH="${MASTER_CSV_PATH}" python - <<'PY'
import csv, re, sys, os
src = os.environ.get("MASTER_MRT_PATH")
dst = os.environ.get("MASTER_CSV_PATH")
if not src or not dst:
    print("ENV not set", file=sys.stderr); sys.exit(1)

# The .mrt has a header block and then a fixed/whitespace table.
rows = []
header = None
with open(src, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # skip comment / blank lines
        if not line.strip(): 
            continue
        if line.lstrip().startswith(("#","|")):
            # Some .mrt files use '|' header separators—just skip
            continue
        # Collapse multiple spaces/tabs into a single comma
        # (SPARC .mrt is consistently whitespace-delimited in the data section)
        parts = re.split(r"\s+", line.strip())
        # Heuristic: first non-comment data line becomes header if it looks textual
        if header is None:
            # If most columns contain alphabetic characters, treat as header
            alpha_cols = sum(any(c.isalpha() for c in p) for p in parts)
            if alpha_cols >= max(2, len(parts)//3):
                header = parts
                rows.append(header)
                continue
            else:
                # No header found; synthesize generic names
                header = [f"col{i+1}" for i in range(len(parts))]
                rows.append(header)
        rows.append(parts)

with open(dst, "w", newline="", encoding="utf-8") as out:
    csv.writer(out).writerows(rows)

print(f">>> [SPARC] Wrote CSV: {dst} (rows={len(rows)})")
PY
echo ">>> [SPARC] Master sheet conversion complete: ${MASTER_CSV_PATH}"

###############################################################################
# 2) Galaxy files to fetch (HIrad + SB), with correct SPARC directory names
#    NOTE: SPARC uses zero-padded IDs in some directory names (e.g., NGC0598, UGC0128)
###############################################################################
# Format per line: "DISPLAY_NAME|DIR_NAME"
GALAXIES=(
  "NGC3198|NGC3198"
  "NGC2403|NGC2403"
  "NGC598|NGC0598"     # M33 (NGC 598) is stored as NGC0598 on SPARC
  "NGC5055|NGC5055"
  "NGC2903|NGC2903"
  "NGC6946|NGC6946"
  "NGC2841|NGC2841"
  "UGC128|UGC0128"     # UGC 128 stored as UGC0128
)

fetch_file() {
  local url="$1"
  local out="$2"
  echo "      -> GET ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${out}" "${url}"
}

echo ">>> [SPARC] Downloading per-galaxy HIrad/SB files ..."
for entry in "${GALAXIES[@]}"; do
  IFS="|" read -r TAG DIRNAME <<< "${entry}"
  echo ">>> [SPARC] Galaxy: ${TAG} (remote dir: ${DIRNAME})"

  # File basenames are usually the same as the directory (e.g., NGC3198_HIrad.dat)
  # For M33 and UGC128 we already mapped to zero-padded DIRNAME, which matches file basenames too.
  HI_URL="${DATA_URL}/${DIRNAME}/${DIRNAME}_HIrad.dat"
  SB_URL="${DATA_URL}/${DIRNAME}/${DIRNAME}_SB.dat"

  HI_OUT="${DEST_DIR}/${TAG}_HIrad.dat"
  SB_OUT="${DEST_DIR}/${TAG}_SB.dat"

  # Try primary URLs
  set +e
  fetch_file "${HI_URL}" "${HI_OUT}"
  status_hi=$?
  fetch_file "${SB_URL}" "${SB_OUT}"
  status_sb=$?
  set -e

  # Fallbacks: occasionally files are named without zero padding in filename even if dir is padded
  if [[ $status_hi -ne 0 ]]; then
    echo "      !! HI file not found at padded path; trying non-padded filename fallback"
    HI_URL_FALLBACK="${DATA_URL}/${DIRNAME}/${TAG}_HIrad.dat"
    fetch_file "${HI_URL_FALLBACK}" "${HI_OUT}"
  fi
  if [[ $status_sb -ne 0 ]]; then
    echo "      !! SB file not found at padded path; trying non-padded filename fallback"
    SB_URL_FALLBACK="${DATA_URL}/${DIRNAME}/${TAG}_SB.dat"
    fetch_file "${SB_URL_FALLBACK}" "${SB_OUT}"
  fi

  # Quick sanity: ensure non-empty files
  for f in "${HI_OUT}" "${SB_OUT}"; do
    if [[ ! -s "${f}" ]]; then
      echo "      !! ERROR: Downloaded file is missing or empty: ${f}"
      echo "         Please open the galaxy page on ${BASE_URL}/data/${DIRNAME}/ to verify filenames."
      exit 2
    fi
  done

  echo ">>> [SPARC] Saved: ${HI_OUT} and ${SB_OUT}"
done

###############################################################################
# 3) Final sanity + friendly next steps
###############################################################################
echo ">>> [SPARC] All downloads complete."
echo ">>> [SPARC] Files now in: ${DEST_DIR}"
echo ">>> [SPARC] Next: run a quick single-galaxy check (density-aware):"
echo ""
echo "    python tools/fit_sparc_er_env.py \\"
echo "      --galaxy_id NGC3198 \\"
echo "      --sparc_dir ${DEST_DIR} \\"
echo "      --out images/sparc_ngc3198_env_fit.png"
echo ""
echo ">>> [SPARC] For batch runs (once you're happy with the first result):"
echo "    python tools/batch_sparc_env_fit.py --sparc_dir ${DEST_DIR} --mode evidence"

