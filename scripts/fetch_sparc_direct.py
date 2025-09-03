#!/usr/bin/env python3
"""
fetch_sparc_direct.py

Directly download SPARC HIrad and SB files for a list of galaxies using
known naming patterns, without relying on directory listing. Suitable for
Windows/PowerShell environments.

Usage:
  python scripts/fetch_sparc_direct.py --dest external_data/Rotmod_LTG \
    --names NGC2403 NGC3198 NGC6503 NGC3621 NGC4258 DDO154 IC2574 UGC128 NGC3741
"""
import argparse
import re
from pathlib import Path
import sys
import time
import requests

BASE_URL = "https://astroweb.cwru.edu/SPARC"
DATA_URL = f"{BASE_URL}/data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DDMM-fetch/1.0)"}

HI_CANDIDATES = [
    "{dir}/{dir}_HIrad.dat",
    "{dir}/{name}_HIrad.dat",
    "{dir}/{dir}_HIrad.txt",
    "{dir}/{name}_HIrad.txt",
    "{dir}/{dir}_HIrad.csv",
    "{dir}/{name}_HIrad.csv",
]
SB_CANDIDATES = [
    "{dir}/{dir}_SB.dat",
    "{dir}/{name}_SB.dat",
    "{dir}/{dir}_SB.txt",
    "{dir}/{name}_SB.txt",
    "{dir}/{dir}_SB.csv",
    "{name}/{name}_SB.csv",
]


def pad_dir_name(name: str) -> str:
    m = re.match(r"^(NGC|UGC|IC)\s*0*([0-9]+)$", name.strip(), re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2)):04d}"
    return name.strip()


def http_download(url: str, dest: Path, retries: int = 2, sleep_s: float = 1.0) -> bool:
    for i in range(1, retries+1):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                print(f"  Saved: {dest} (from {url})")
                return True
            else:
                print(f"  HTTP {r.status_code} for {url} (attempt {i}/{retries})")
        except Exception as e:
            print(f"  Error GET {url} (attempt {i}/{retries}): {e}")
        time.sleep(sleep_s)
    return False


def fetch_one(gal: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    dir_p = pad_dir_name(gal)
    ok = {"HI": None, "SB": None}

    # Try HI candidates
    for pat in HI_CANDIDATES:
        rel = pat.format(dir=dir_p, name=gal)
        url = f"{DATA_URL}/{rel}"
        out = dest / f"{gal}_HIrad{Path(rel).suffix.lower()}"
        if http_download(url, out):
            ok["HI"] = str(out)
            break

    # Try SB candidates
    for pat in SB_CANDIDATES:
        rel = pat.format(dir=dir_p, name=gal)
        url = f"{DATA_URL}/{rel}"
        out = dest / f"{gal}_SB{Path(rel).suffix.lower()}"
        if http_download(url, out):
            ok["SB"] = str(out)
            break

    return ok


def main():
    ap = argparse.ArgumentParser(description="Direct-fetch SPARC HIrad/SB by known filename patterns")
    ap.add_argument("--dest", required=True, help="Destination directory")
    ap.add_argument("--names", nargs="+", help="Galaxy names (e.g., NGC3198 UGC128)")
    args = ap.parse_args()

    dest = Path(args.dest)
    summary = []
    for g in args.names:
        print(f"Fetching {g} …")
        res = fetch_one(g, dest)
        summary.append((g, res["HI"], res["SB"]))

    print("\nSummary:")
    got = 0
    for g, hi, sb in summary:
        print(f"  {g}: HI={'OK' if hi else '—'}, SB={'OK' if sb else '—'}")
        got += int(bool(hi)) + int(bool(sb))
    print(f"Downloaded files: {got} (HI+SB) for {len(summary)} galaxies")


if __name__ == "__main__":
    sys.exit(main())

