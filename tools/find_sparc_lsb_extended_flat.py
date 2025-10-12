#!/usr/bin/env python3
"""
Find low-surface-brightness SPARC galaxies with extended, flat outer rotation curves (UGC 128–like).

- Parses MasterSheet_SPARC.csv under --sparc-dir robustly (skipping MRT-style preamble)
- Scans *_rotmod.dat to compute:
  - Rmax_kpc and extent ratio Rmax/Rdisk
  - Outer flatness via linear slope on the last 25% of radii and fractional spread
- Filters by (defaults chosen to capture UGC 128–like systems):
  - SBdisk0_Lsun_pc2 <= 120 (low central SB at 3.6µm)
  - Rmax/Rdisk >= 8 (extended radial coverage)
  - |outer_slope| <= 1.5 km/s/kpc and outer fractional range <= 0.15
  - Quality flag Q <= 2 (optional)

Outputs a ranked CSV and prints a concise summary.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path for local imports
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# Reuse existing simple rotmod loader
from data_loaders.sparc_loader import load_rotmod

# Column schema we will enforce when parsing MasterSheet_SPARC.csv (MRT-converted CSV)
MASTER_COLS = [
    'Name','T','D_Mpc','e_D','f_D','Inc','e_Inc','L_3p6_1e9Lsun','e_L_3p6',
    'Reff_kpc','SBeff_Lsun_pc2','Rdisk_kpc','SBdisk0_Lsun_pc2','MHI_1e9Msun',
    'RHI_kpc','Vflat_kms','e_Vflat_kms','Q','Ref'
]


@dataclass
class GalaxyMeta:
    name: str
    Rd_kpc: Optional[float]
    SB0_Lsun_pc2: Optional[float]
    Q: Optional[int]
    Vflat_kms: Optional[float]
    RHI_kpc: Optional[float]


def standardize_id(gid: str) -> str:
    # Uppercase, remove spaces, and zero-pad common prefixes to 4 digits for matching
    g = re.sub(r"\s+", "", gid.strip().upper())
    m = re.match(r"^(NGC|UGC|IC)(\d+)$", g)
    if m:
        return f"{m.group(1)}{int(m.group(2)):04d}"
    return g


def parse_master_csv(sparc_dir: Path) -> Dict[str, GalaxyMeta]:
    ms_path = sparc_dir / "MasterSheet_SPARC.csv"
    if not ms_path.exists():
        raise FileNotFoundError(f"Master sheet not found: {ms_path}")

    # Read lines and find the first data row (heuristic: 19 comma-separated fields, with numeric in cols 2 and 3)
    lines = ms_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = None
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) >= 19:
            # Test numeric-like for T (int) and D (float)
            try:
                int(parts[1])
                float(parts[2])
                start_idx = i
                break
            except Exception:
                continue
    if start_idx is None:
        raise ValueError("Could not locate data section in MasterSheet_SPARC.csv")

    # Parse into DataFrame with enforced columns
    df = pd.read_csv(ms_path, skiprows=start_idx, header=None, names=MASTER_COLS)

    meta: Dict[str, GalaxyMeta] = {}
    for _, r in df.iterrows():
        name = str(r['Name']).strip()
        if not name or name.lower() == 'nan':
            continue
        key = standardize_id(name)
        def fget(col):
            try:
                v = float(r[col])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
            return None
        def iget(col):
            try:
                v = int(r[col])
                return v
            except Exception:
                return None
        meta[key] = GalaxyMeta(
            name=name,
            Rd_kpc=fget('Rdisk_kpc'),
            SB0_Lsun_pc2=fget('SBdisk0_Lsun_pc2'),
            Q=iget('Q'),
            Vflat_kms=fget('Vflat_kms'),
            RHI_kpc=fget('RHI_kpc'),
        )
    return meta


def compute_outer_metrics(R: np.ndarray, V: np.ndarray, frac_tail: float = 0.25) -> Tuple[float, float, float, int]:
    """
    Compute simple outer flatness metrics on the last `frac_tail` of the radial extent.
    Returns: (slope_kms_per_kpc, frac_range, V_mean, n_tail)
    """
    if R.size < 5:
        return (np.nan, np.nan, np.nan, int(R.size))
    # Tail selection
    n = R.size
    start_idx = max(0, int(n * (1 - frac_tail)))
    Rt = R[start_idx:]
    Vt = V[start_idx:]
    # Guard for degenerate Rt
    if np.max(Rt) - np.min(Rt) < 1e-6:
        return (np.nan, np.nan, float(np.mean(Vt)), int(Vt.size))
    # Weighted linear fit (uniform weights, rotmod eV not always reliable across files)
    try:
        p = np.polyfit(Rt, Vt, deg=1)
        slope = float(p[0])
    except Exception:
        slope = np.nan
    vmean = float(np.mean(Vt)) if Vt.size else np.nan
    frange = float((np.max(Vt) - np.min(Vt)) / vmean) if (Vt.size and vmean > 1e-6) else np.nan
    return (slope, frange, vmean, int(Vt.size))


def scan_galaxies(sparc_dir: Path,
                  sb_max: float,
                  r_over_rd_min: float,
                  slope_abs_max: float,
                  outer_frac_range_max: float,
                  qmax: Optional[int]) -> List[dict]:
    # Load metadata
    meta = parse_master_csv(sparc_dir)

    # Find rotmod files
    rotmods = sorted(sparc_dir.glob("*_rotmod.dat"))
    out_rows: List[dict] = []

    for p in rotmods:
        gid_raw = p.stem.replace('_rotmod', '')
        gid = standardize_id(gid_raw)
        try:
            dat = load_rotmod(str(p))
        except Exception:
            continue
        R = np.array(dat['R_kpc'], dtype=float)
        V = np.array(dat['Vobs_kms'], dtype=float)
        if R.size == 0:
            continue
        Rmax = float(np.max(R))
        m = meta.get(gid)
        Rd = m.Rd_kpc if m else None
        SB0 = m.SB0_Lsun_pc2 if m else None
        Q = m.Q if m else None
        Vflat = m.Vflat_kms if m else None
        RHI = m.RHI_kpc if m else None

        r_over_rd = (Rmax / Rd) if (Rd and Rd > 0) else np.nan
        slope, frange, vmean, n_tail = compute_outer_metrics(R, V)

        # Filter flags
        pass_lsb = (SB0 is not None) and (SB0 <= sb_max)
        pass_extent = (not np.isnan(r_over_rd)) and (r_over_rd >= r_over_rd_min)
        pass_flat = (not np.isnan(slope)) and (abs(slope) <= slope_abs_max) and (not np.isnan(frange)) and (frange <= outer_frac_range_max)
        pass_q = (Q is None) or (qmax is None) or (Q <= qmax)

        passed = pass_lsb and pass_extent and pass_flat and pass_q

        out_rows.append({
            'galaxy': gid,
            'name': (m.name if m else gid_raw),
            'Q': Q,
            'SBdisk0_Lsun_pc2': SB0,
            'Rdisk_kpc': Rd,
            'RHI_kpc': RHI,
            'Rmax_kpc': Rmax,
            'Rmax_over_Rd': r_over_rd,
            'Vflat_kms': Vflat,
            'V_mean_outer_kms': vmean,
            'outer_slope_kms_per_kpc': slope,
            'outer_frac_range': frange,
            'n_points_tail': n_tail,
            'rotmod_path': str(p),
            'passed': passed,
            'flags': {
                'lsb': pass_lsb,
                'extent': pass_extent,
                'flat': pass_flat,
                'q_ok': pass_q,
            }
        })

    # Rank: primarily by Rmax/Rd desc, then by SB0 asc, then by |slope|
    def sort_key(r):
        rrd = r.get('Rmax_over_Rd')
        sb = r.get('SBdisk0_Lsun_pc2')
        sl = abs(r.get('outer_slope_kms_per_kpc') or np.nan)
        return (
            -9999 if (rrd is None or np.isnan(rrd)) else -rrd,
            9999 if (sb is None or np.isnan(sb)) else sb,
            9999 if (sl is None or np.isnan(sl)) else sl,
        )

    out_rows.sort(key=sort_key)
    return out_rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten flags for CSV
    rows2 = []
    for r in rows:
        f = r.pop('flags', {})
        r2 = dict(r)
        r2['flag_lsb'] = f.get('lsb')
        r2['flag_extent'] = f.get('extent')
        r2['flag_flat'] = f.get('flat')
        r2['flag_q_ok'] = f.get('q_ok')
        rows2.append(r2)
    if rows2:
        cols = list(rows2[0].keys())
    else:
        cols = [
            'galaxy','name','Q','SBdisk0_Lsun_pc2','Rdisk_kpc','RHI_kpc','Rmax_kpc','Rmax_over_Rd',
            'Vflat_kms','V_mean_outer_kms','outer_slope_kms_per_kpc','outer_frac_range','n_points_tail',
            'rotmod_path','passed','flag_lsb','flag_extent','flag_flat','flag_q_ok'
        ]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows2:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Find SPARC LSB galaxies with extended, flat outer rotation curves")
    ap.add_argument('--sparc-dir', default='external_data/Rotmod_LTG', help='Directory containing *_rotmod.dat and MasterSheet_SPARC.csv')
    ap.add_argument('--sb-max', type=float, default=120.0, help='Max central SBdisk0 (Lsun/pc^2) to qualify as LSB (default: 120)')
    ap.add_argument('--r-over-rd-min', type=float, default=8.0, help='Minimum extent Rmax/Rdisk (default: 8)')
    ap.add_argument('--slope-abs-max', type=float, default=1.5, help='Max |outer slope| (km/s/kpc) over last 25% of points (default: 1.5)')
    ap.add_argument('--outer-frac-range-max', type=float, default=0.15, help='Max fractional range (max-min)/mean in the outer segment (default: 0.15)')
    ap.add_argument('--qmax', type=int, default=2, help='Maximum SPARC quality flag Q to include (default: 2)')
    ap.add_argument('--output', default='results/lsb_extended_flat_candidates.csv', help='Output CSV path')
    args = ap.parse_args()

    sparc_dir = Path(args.sparc_dir)
    rows = scan_galaxies(
        sparc_dir=sparc_dir,
        sb_max=args.sb_max,
        r_over_rd_min=args.r_over_rd_min,
        slope_abs_max=args.slope_abs_max,
        outer_frac_range_max=args.outer_frac_range_max,
        qmax=args.qmax,
    )

    # Keep only those passing all filters for summary print; write full table to CSV
    write_csv(rows, Path(args.output))

    passing = [r for r in rows if r['passed']]
    print(f"Total galaxies scanned: {len(rows)}")
    print(f"Candidates (passed all filters): {len(passing)}")
    print("Top candidates (up to 20):")
    for r in passing[:20]:
        print(
            f"- {r['name']} ({r['galaxy']}), Q={r['Q']}, SB0={r['SBdisk0_Lsun_pc2']:.2f}, "
            f"Rmax/Rd={(r['Rmax_over_Rd'] or float('nan')):.2f}, slope={r['outer_slope_kms_per_kpc']:.2f} km/s/kpc, "
            f"outer ΔV/V≈{r['outer_frac_range']:.2f}"
        )
    # Sanity ping for UGC 128 if present
    for r in passing:
        if standardize_id(r['galaxy']) in ("UGC0128", "UGC0128" ) or r['name'].upper().startswith("UGC00128"):
            print("Note: UGC 128 appears in candidates.")
            break


if __name__ == '__main__':
    main()