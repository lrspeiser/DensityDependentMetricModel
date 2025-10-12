#!/usr/bin/env python3
"""
convert_castles_small_to_rar_lenses.py

Purpose
- Convert a CASTLES-like lens list into the CSV format expected by
  scripts/next_steps_from_run.py --lensing-sample-csv.
- Uses documented assumptions to derive stellar mass and size when only
  image separation and velocity dispersion are available.

Input CSV
- Supports two formats:
  (A) Proper CSV with headers like: id,lens_name,G,zs,zl,...,Nim,size_arcsec,dt_days,sigma_km_s
  (B) The older "small" table exported as messy CSV-like text; for this we fall back to regex.

Output CSV (clean)
- lens_id,z_l,z_s,log10M_star,Re_kpc,n_sersic,theta_E_obs_arcsec

Assumptions (Option B, documented)
- θ_E_obs_arcsec is estimated from size_arcsec and the image code on the line:
  - If the code indicates a ring (contains 'ER' or e.g. '2R','4R'): θ_E_obs ≈ size_arcsec
  - Else (doubles/quads): θ_E_obs ≈ size_arcsec / 2
- Stellar mass from a Faber–Jackson-like scaling around SLACS-like ETGs:
  - log10 M_star = 11.20 + 4.0 * log10(σ / 225 km/s)
- Effective radius scaling with σ:
  - Re_kpc ≈ 4.5 * (σ / 225 km/s)^1.2
- n_sersic defaults to 4 (de Vaucouleurs-like massive ETGs)

Notes
- First attempts strict CSV parsing via csv.DictReader. If expected headers are missing,
  or nothing parsable is found, falls back to the legacy regex path.
- You can restrict to a selected set of lenses via --select; otherwise the script will
  try to parse all and include only those with z_s,z_l,size_arcsec, and σ.

"""
import argparse
import re
import sys
import math
import csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Convert CASTLES-like CSV to orchestrator lens CSV")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to input CASTLES CSV")
    ap.add_argument("--out", dest="out_path", required=True, help="Path to output CSV for orchestrator")
    ap.add_argument("--select", nargs="*", default=None, help="Optional list of lens_name to include; default tries all parsable")
    return ap.parse_args()


def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None


def extract_with_regex(line: str, lens_name: str):
    """Legacy fallback for messy CASTLES small-table text lines."""
    # Extract z_s and z_l: first two floats following the grade letter after lens name
    m = re.search(r"^\s*\d+\s*,\s*" + re.escape(lens_name) + r"\s+[A-Z]\s+([0-9.]+)\s+([0-9.]+)", line)
    if not m:
        return None
    z_s = safe_float(m.group(1))
    z_l = safe_float(m.group(2))
    if z_s is None or z_l is None:
        return None

    # Extract sigma (km/s): take the last occurrence of a "num±num" pattern and use the first num
    sig_pairs = re.findall(r"([0-9]+(?:\.[0-9]+)?)±([0-9]+(?:\.[0-9]+)?)", line)
    sigma = None
    if sig_pairs:
        sigma = safe_float(sig_pairs[-1][0])

    # Extract size_arcsec: choose the largest float within (0.6, 8.0) — avoids magnitudes and tiny E_BV
    floats = [safe_float(x) for x in re.findall(r"(?<!\d)(\d+\.\d+)", line)]
    floats = [v for v in floats if v is not None]
    size_candidates = [v for v in floats if 0.6 <= v <= 8.0 and abs(v - z_s) > 1e-3 and abs(v - z_l) > 1e-3]
    size_arcsec = max(size_candidates) if size_candidates else None

    # Ring detection via tokens like 'ER' or 'R' in image code fields
    is_ring = ("ER" in line) or (re.search(r"\b[2-9]?R\b", line) is not None)

    return {
        "z_s": z_s,
        "z_l": z_l,
        "sigma": sigma,
        "size_arcsec": size_arcsec,
        "is_ring": is_ring,
    }


def derive_mass_size(sigma_kms: float):
    # Faber–Jackson-like mass scaling
    if sigma_kms <= 0:
        return None, None
    log10M = 11.20 + 4.0 * math.log10(sigma_kms / 225.0)
    Re_kpc = 4.5 * ((sigma_kms / 225.0) ** 1.2)
    return log10M, Re_kpc


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect selected lens names
    selected = set([s.strip() for s in args.select]) if args.select else None

    rows = []
    parsed_csv = False

    # Attempt strict CSV parsing first
    try:
        with in_path.open("r", encoding="utf-8", errors="ignore") as f:
            rdr = csv.DictReader(f)
            fieldnames = [fn.strip() for fn in (rdr.fieldnames or [])]
            # Require minimal set of recognizable headers
            need_any = {"lens_name", "zs", "zl"}
            if fieldnames and need_any.issubset(set(fieldnames)):
                parsed_csv = True
                for row in rdr:
                    if not row:
                        continue
                    lens = (row.get("lens_name") or row.get("lens") or row.get("id") or "").strip()
                    if not lens:
                        continue
                    if selected and lens not in selected:
                        continue
                    # Extract essential fields
                    z_s = safe_float((row.get("zs") or row.get("z_s") or "").strip())
                    z_l = safe_float((row.get("zl") or row.get("z_l") or "").strip())
                    size_str = (row.get("size_arcsec") or row.get("sep_arcsec") or "").strip()
                    size_arcsec = safe_float(size_str)
                    nim = (row.get("Nim") or row.get("nim") or "").strip()
                    sig_str = (row.get("sigma_km_s") or row.get("sigma") or "").strip()
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*±\s*([0-9]+(?:\.[0-9]+)?)", sig_str)
                    sigma = safe_float(m.group(1)) if m else None
                    # Determine ring vs non-ring
                    is_ring = ("ER" in nim) or (re.search(r"\b[2-9]?R\b", nim) is not None)
                    if z_s is None or z_l is None or size_arcsec is None or sigma is None:
                        continue
                    theta_obs = size_arcsec if is_ring else (size_arcsec / 2.0)
                    log10M, Re_kpc = derive_mass_size(sigma)
                    if log10M is None or Re_kpc is None:
                        continue
                    rows.append({
                        "lens_id": lens,
                        "z_l": z_l,
                        "z_s": z_s,
                        "log10M_star": log10M,
                        "Re_kpc": Re_kpc,
                        "n_sersic": 4,
                        "theta_E_obs_arcsec": theta_obs,
                    })
    except Exception:
        parsed_csv = False
        # fall through to regex path

    # Fallback to legacy regex parsing if needed
    if not rows:
        with in_path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()  # discard header
            for raw in f:
                if not raw.strip():
                    continue
                mname = re.match(r"\s*\d+\s*,\s*([^,\t]+)", raw)
                if not mname:
                    continue
                lens = mname.group(1).strip()
                if selected and lens not in selected:
                    continue
                info = extract_with_regex(raw, lens)
                if not info:
                    continue
                z_s = info["z_s"]; z_l = info["z_l"]
                sigma = info["sigma"]; size_arcsec = info["size_arcsec"]; is_ring = info["is_ring"]
                if z_s is None or z_l is None or size_arcsec is None or sigma is None:
                    continue
                theta_obs = size_arcsec if is_ring else (size_arcsec / 2.0)
                log10M, Re_kpc = derive_mass_size(sigma)
                if log10M is None or Re_kpc is None:
                    continue
                rows.append({
                    "lens_id": lens,
                    "z_l": z_l,
                    "z_s": z_s,
                    "log10M_star": log10M,
                    "Re_kpc": Re_kpc,
                    "n_sersic": 4,
                    "theta_E_obs_arcsec": theta_obs,
                })

    if not rows:
        print("No parsable lenses found. Check input format.", file=sys.stderr)
        sys.exit(2)

    # Write output CSV
    with out_path.open("w", encoding="utf-8") as g:
        g.write("lens_id,z_l,z_s,log10M_star,Re_kpc,n_sersic,theta_E_obs_arcsec\n")
        for r in rows:
            g.write(f"{r['lens_id']},{r['z_l']},{r['z_s']},{r['log10M_star']:.4f},{r['Re_kpc']:.4f},{r['n_sersic']},{r['theta_E_obs_arcsec']:.4f}\n")

    print(f"Wrote {len(rows)} lens rows -> {out_path}")


if __name__ == "__main__":
    main()

