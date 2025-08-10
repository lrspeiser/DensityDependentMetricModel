#!/usr/bin/env python3
"""
fetch_sparc_hirad_sb_v2.py

Robust SPARC fetcher that:
  1) Downloads the SPARC master sheet (.mrt) and converts to CSV.
  2) Crawls each galaxy's directory on the SPARC site to discover actual filenames.
  3) Grabs HIrad and SB files with flexible matching (case-insensitive, .dat/.txt/.csv).
  4) (Optional) Also fetches Rotmod component files (useful fallback for v_bar).
  5) Never aborts on first failure; emits a final summary (downloaded / missing).

Console logging is explicit so you can paste it back to me if anything looks off.
"""

import argparse
import csv
import os
import re
import sys
import time
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import requests

BASE_URL = "https://astroweb.cwru.edu/SPARC"
DATA_URL = f"{BASE_URL}/data"
MASTER_MRT_URL = "https://zenodo.org/records/16284118/files/SPARC_Lelli2016c.mrt"
MASSMODELS_MRT_URL = "https://zenodo.org/records/16284118/files/MassModels_Lelli2016c.mrt"
ROTMOD_ZIP_URL = "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip"
BUNDLE_ZIP_URL = "https://zenodo.org/records/16284118/files/sparc_database.zip"  # optional mirror/bundle

UA = {"User-Agent": "Mozilla/5.0 (compatible; ER-fetch/1.0)"}

# File patterns we’ll consider “good enough”
HI_PATTERNS = [
    r"_HIrad\.dat$", r"_HIrad\.txt$", r"_HIrad\.csv$",
    r"_HI[_-]?rad.*\.dat$", r"_HI[_-]?rad.*\.txt$", r"_HI[_-]?rad.*\.csv$",
]
SB_PATTERNS = [
    r"_SB\.dat$", r"_SB\.txt$", r"_SB\.csv$",
    r"_SB[_-]?prof.*\.dat$", r"_SB[_-]?prof.*\.txt$", r"_SB[_-]?prof.*\.csv$",
]
# Common Rotmod component files (optional)
ROTMOD_PATTERNS = [
    r"_rotmod\.dat$", r"_rotmod\.txt$",
    r"_HIrotmod\.dat$", r"_gasrotmod\.dat$", r"_strotmod\.dat$",
]

def log(msg: str) -> None:
    print(f">>> [SPARC] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f">>> [WARN ] {msg}", flush=True)

def err(msg: str) -> None:
    print(f">>> [ERROR] {msg}", file=sys.stderr, flush=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def http_get(url: str, timeout: float = 30.0) -> requests.Response:
    return requests.get(url, headers=UA, timeout=timeout)

def http_download(url: str, dest: Path, retries: int = 3, sleep_s: float = 1.5) -> bool:
    for i in range(1, retries+1):
        try:
            r = http_get(url)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return True
            warn(f"GET {url} -> HTTP {r.status_code} (attempt {i}/{retries})")
        except Exception as e:
            warn(f"GET {url} failed (attempt {i}/{retries}): {e}")
        time.sleep(sleep_s)
    return False

def pad_dir_name(name: str) -> str:
    """SPARC directories often zero-pad NGC/UGC/IC to 4 digits."""
    m = re.match(r'^(NGC|UGC|IC)\s*0*([0-9]+)$', name.strip(), re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2)):04d}"
    return name.strip()

def scrape_links(index_html: str) -> List[str]:
    """Extract href targets from a simple directory index page."""
    # Very simple href parser (no external deps)
    links = re.findall(r'href=["\']([^"\']+)["\']', index_html, flags=re.IGNORECASE)
    return [unescape(h) for h in links]

def list_remote_files(dir_url: str) -> List[str]:
    """Return a list of candidate file names (not full URLs) in a remote directory."""
    try:
        r = http_get(dir_url)
        if r.status_code != 200:
            warn(f"LIST {dir_url} -> HTTP {r.status_code}")
            return []
        links = scrape_links(r.text)
        # Keep only files in this directory (no parent/anchors)
        files = []
        for href in links:
            if href in ("../", "./"):
                continue
            # Directory listings often show plain filenames
            name = href.split("/")[-1]
            # Skip subdirs (end with '/')
            if not name or name.endswith("/"):
                continue
            files.append(name)
        return files
    except Exception as e:
        warn(f"Could not list {dir_url}: {e}")
        return []

def pick_file(files: List[str], patterns: List[str]) -> Optional[str]:
    """Pick first filename that matches any of the regex patterns (case-insensitive)."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for f in files:
            if rx.search(f):
                return f
    return None

def mrt_to_csv(mrt_path: Path, csv_path: Path) -> int:
    """Convert .mrt table (whitespace) to CSV."""
    rows = []
    header = None
    with mrt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("|"):
                continue
            parts = re.split(r"\s+", s)
            if header is None:
                # crude header detection
                if sum(any(c.isalpha() for c in p) for p in parts) >= max(2, len(parts)//3):
                    header = parts
                else:
                    header = [f"col{i+1}" for i in range(len(parts))]
                rows.append(header)
            rows.append(parts)
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        csv.writer(out).writerows(rows)
    return len(rows)

def parse_names_from_master(csv_path: Path) -> List[str]:
    names: List[str] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            # try common name column
            idx = None
            for i, h in enumerate(header):
                if h.strip().lower() in ("name", "gal", "galaxy", "object"):
                    idx = i; break
            if idx is None:
                idx = 0
            for row in reader:
                if not row or len(row) <= idx: continue
                nm = re.sub(r"\s+", "", row[idx].strip())
                if nm:
                    names.append(nm)
    except Exception as e:
        warn(f"Failed to parse names from {csv_path.name}: {e}")
    # de-dup
    seen = set(); out = []
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    if not out:
        out = ["NGC3198", "NGC2403", "NGC598", "NGC5055", "NGC2903", "NGC6946", "NGC2841", "UGC128"]
        warn("Using fallback name list.")
    return out

def download_for_gal(dest: Path, gal: str, fetch_rotmod: bool=False) -> Dict[str, Optional[str]]:
    """
    Try to fetch HIrad and SB (and optional Rotmod component files) for a galaxy.
    Returns dict with keys: 'HI', 'SB', 'ROTMOD' (values are local filepaths or None).
    """
    results = {"HI": None, "SB": None, "ROTMOD": None}
    dir_padded = pad_dir_name(gal)
    dir_url = f"{DATA_URL}/{dir_padded}/"
    log(f"Galaxy {gal}: listing {dir_url}")
    files = list_remote_files(dir_url)
    if not files:
        warn(f"No listing for {gal} at {dir_url}")
        return results

    # pick candidates
    hi = pick_file(files, HI_PATTERNS)
    sb = pick_file(files, SB_PATTERNS)
    rot = pick_file(files, ROTMOD_PATTERNS) if fetch_rotmod else None

    # Download each if found
    def dl(fname: str, tag: str) -> Optional[str]:
        url = f"{dir_url}{fname}"
        out = dest / f"{gal}_{tag}{Path(fname).suffix.lower()}"
        if http_download(url, out):
            log(f"  Saved {tag}: {out.name}")
            return str(out)
        warn(f"  Could not download {tag} from {url}")
        return None

    if hi:   results["HI"] = dl(hi, "HIrad")
    else:    warn(f"  No HIrad-like file found for {gal} in {dir_url}")
    if sb:   results["SB"] = dl(sb, "SB")
    else:    warn(f"  No SB-like file found for {gal} in {dir_url}")
    if rot:  results["ROTMOD"] = dl(rot, "rotmod")

    return results

def main():
    ap = argparse.ArgumentParser(description="Robust SPARC HIrad/SB fetcher with directory crawling.")
    ap.add_argument("--dest", required=True, help="Destination directory (e.g., external_data/Rotmod_LTG)")
    ap.add_argument("--names", nargs="*", default=None, help="Optional subset of galaxy names")
    ap.add_argument("--rotmod", action="store_true", help="Also try to fetch Rotmod component files")
    ap.add_argument("--dry-run", action="store_true", help="List what would be fetched without downloading")
    ap.add_argument("--use-zenodo", action="store_true", help="Download Rotmod_LTG.zip and MRTs from Zenodo and extract")
    args = ap.parse_args()

    dest = Path(args.dest)
    ensure_dir(dest)
    log(f"Target directory: {dest}")

    # Master sheet (Zenodo)
    mrt = dest / "MasterSheet_SPARC.mrt"
    csvp = dest / "MasterSheet_SPARC.csv"
    if not csvp.exists():
        log("Fetching master sheet (.mrt) from Zenodo…")
        ok = http_download(MASTER_MRT_URL, mrt)
        if not ok:
            err("Failed to download the master .mrt file from Zenodo."); sys.exit(2)
        log(f"Saved: {mrt}")
        log("Converting .mrt -> .csv …")
        rows = mrt_to_csv(mrt, csvp)
        log(f"Wrote CSV: {csvp} (rows={rows})")
    else:
        log(f"Master sheet already present: {csvp.name}")

    # Mass models table (optional, for reference)
    mm_mrt = dest / "MassModels_Lelli2016c.mrt"
    if not mm_mrt.exists():
        log("Fetching MassModels_Lelli2016c.mrt from Zenodo…")
        _ = http_download(MASSMODELS_MRT_URL, mm_mrt)
        if mm_mrt.exists():
            log(f"Saved: {mm_mrt}")

    # Galaxy list
    if args.names:
        galaxies = [re.sub(r"\s+", "", g) for g in args.names if g.strip()]
        log(f"Using user-specified list ({len(galaxies)}).")
    else:
        galaxies = parse_names_from_master(csvp)
        log(f"Parsed {len(galaxies)} names from master sheet.")

    # Optionally fetch Rotmod zip from Zenodo
    if args.use_zenodo:
        import zipfile, io
        rotmod_zip = dest / "Rotmod_LTG.zip"
        log("Downloading Rotmod_LTG.zip from Zenodo…")
        ok = http_download(ROTMOD_ZIP_URL, rotmod_zip)
        if ok:
            log(f"Saved: {rotmod_zip}")
            log("Extracting Rotmod_LTG.zip …")
            try:
                with zipfile.ZipFile(rotmod_zip, 'r') as zf:
                    zf.extractall(dest)
                log(f"Extracted to: {dest}")
            except Exception as e:
                warn(f"Failed to extract Rotmod_LTG.zip: {e}")
        else:
            warn("Failed to download Rotmod_LTG.zip from Zenodo.")

    # Dry-run just lists available matches
    if args.dry_run:
        log("[DRY-RUN] Inspecting a few galaxies for file presence …")
        for gal in galaxies[:10]:
            durl = f"{DATA_URL}/{pad_dir_name(gal)}/"
            files = list_remote_files(durl)
            log(f"{gal}: {len(files)} items at {durl}")
            hi = pick_file(files, HI_PATTERNS)
            sb = pick_file(files, SB_PATTERNS)
            rot = pick_file(files, ROTMOD_PATTERNS) if args.rotmod else None
            log(f"  picks -> HI={hi}, SB={sb}, ROTMOD={rot}")
        log("[DRY-RUN] Done.")
        return

    # Download loop (does NOT stop on first failure)
    summary = []
    for gal in galaxies:
        res = download_for_gal(dest, gal, fetch_rotmod=args.rotmod)
        summary.append((gal, res["HI"], res["SB"], res["ROTMOD"]))

    # Write a summary CSV
    sumcsv = dest / "download_summary.csv"
    with sumcsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["galaxy", "HIrad_local", "SB_local", "ROTMOD_local"])
        for gal, hi, sb, rot in summary:
            w.writerow([gal, hi or "", sb or "", rot or ""])
    log(f"Wrote summary: {sumcsv}")

    # Final tallies
    got_hi = sum(1 for _, hi, _, _ in summary if hi)
    got_sb = sum(1 for _, _, sb, _ in summary if sb)
    log(f"Done. HIrad files: {got_hi}/{len(summary)}  |  SB files: {got_sb}/{len(summary)}")
    miss = [(gal, hi, sb) for gal, hi, sb, _ in summary if not (hi and sb)]
    if miss:
        warn("Some galaxies are missing HIrad or SB. Check download_summary.csv; "
             "you can re-run with --names to focus on the missing set.")

if __name__ == "__main__":
    main()

