#!/usr/bin/env python3
"""
fetch_sparc.py
Purpose: Download SPARC master sheet + per-galaxy HIrad/SB files into the expected directory.
Usage:   python scripts/fetch_sparc.py external_data/Rotmod_LTG
Notes:
 - Handles zero-padded directory names used by SPARC (e.g., NGC0598 for M33, UGC0128).
 - Converts the SPARC machine-readable table (.mrt) to CSV for the loader.
 - Emits clear console logs at each step.
"""
from __future__ import annotations
import sys
import os
import re
import csv
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_URL = "https://astroweb.cwru.edu/SPARC"
DATA_URL = f"{BASE_URL}/data"
MASTER_MRT_URL = f"{BASE_URL}/SPARC_Lelli2016c.mrt"

# (display_name, dir_name)
GALAXIES = [
    ("NGC3198", "NGC3198"),
    ("NGC2403", "NGC2403"),
    ("NGC598",  "NGC0598"),   # M33
    ("NGC5055", "NGC5055"),
    ("NGC2903", "NGC2903"),
    ("NGC6946", "NGC6946"),
    ("NGC2841", "NGC2841"),
    ("UGC128",  "UGC0128"),
]


def http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def main():
    dest_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("external_data/Rotmod_LTG")
    print(f">>> [SPARC] Target directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) Master sheet
    master_mrt_path = dest_dir / "MasterSheet_SPARC.mrt"
    master_csv_path = dest_dir / "MasterSheet_SPARC.csv"

    print(">>> [SPARC] Fetching master sheet (.mrt) ...")
    data = http_get(MASTER_MRT_URL)
    master_mrt_path.write_bytes(data)
    print(f">>> [SPARC] Saved: {master_mrt_path}")

    print(">>> [SPARC] Converting master sheet .mrt -> .csv (for your loader) ...")
    rows: list[list[str]] = []
    header = None
    for line in master_mrt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith(("#", "|")):
            continue
        parts = re.split(r"\s+", line.strip())
        if header is None:
            alpha_cols = sum(any(c.isalpha() for c in p) for p in parts)
            if alpha_cols >= max(2, len(parts)//3):
                header = parts
                rows.append(header)
                continue
            else:
                header = [f"col{i+1}" for i in range(len(parts))]
                rows.append(header)
        rows.append(parts)
    with master_csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f">>> [SPARC] Wrote CSV: {master_csv_path} (rows={len(rows)})")

    # 2) Per-galaxy HIrad/SB
    print(">>> [SPARC] Downloading per-galaxy HIrad/SB files ...")
    successes = {}
    for display, dirname in GALAXIES:
        print(f">>> [SPARC] Galaxy: {display} (remote dir: {dirname})")
        hi_url = f"{DATA_URL}/{dirname}/{dirname}_HIrad.dat"
        sb_url = f"{DATA_URL}/{dirname}/{dirname}_SB.dat"
        hi_out = dest_dir / f"{display}_HIrad.dat"
        sb_out = dest_dir / f"{display}_SB.dat"

        def try_fetch(url: str, out: Path) -> bool:
            try:
                print(f"      -> GET {url}")
                out.write_bytes(http_get(url))
                return True
            except Exception as e:
                print(f"      !! fetch failed: {e}")
                return False

        ok_hi = try_fetch(hi_url, hi_out)
        ok_sb = try_fetch(sb_url, sb_out)

        # Fallback: non-padded filename if initial failed
        if not ok_hi:
            hi_url_fallback = f"{DATA_URL}/{dirname}/{display}_HIrad.dat"
            ok_hi = try_fetch(hi_url_fallback, hi_out)
        if not ok_sb:
            sb_url_fallback = f"{DATA_URL}/{dirname}/{display}_SB.dat"
            ok_sb = try_fetch(sb_url_fallback, sb_out)

        # Sanity check non-empty
        for f in (hi_out, sb_out):
            if not f.exists() or f.stat().st_size == 0:
                print(f"      !! ERROR: Downloaded file is missing or empty: {f}")
                raise SystemExit(2)

        print(f">>> [SPARC] Saved: {hi_out} and {sb_out}")
        successes[display] = {
            "HI": str(hi_out),
            "SB": str(sb_out),
        }

    (dest_dir / "fetch_sparc_manifest.json").write_text(json.dumps(successes, indent=2), encoding="utf-8")

    print(">>> [SPARC] All downloads complete.")
    print(f">>> [SPARC] Files now in: {dest_dir}")
    print(">>> [SPARC] Next: run a quick single-galaxy check (density-aware):")
    print("    python tools/fit_sparc_er_env.py")
    print("      --galaxy_id NGC3198")
    print(f"      --sparc_dir {dest_dir}")
    print("      --out images/sparc_ngc3198_env_fit.png")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr); sys.exit(1)
    except URLError as e:
        print(f"URL error: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

