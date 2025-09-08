#!/usr/bin/env python3
"""
estimate_lens_mstar_re.py — helper to estimate log10(M_star) from an apparent magnitude
and M/L, and to convert an effective radius Re (arcsec) to kpc at the lens redshift.

Optionally, it can update docs/lensing_targets.csv in-place for a given lens_id.

Examples
1) Compute values only (I-band, M/L_I=3):
   python scripts/estimate_lens_mstar_re.py \
     --lens-id Q2237+0305 \
     --z_l 0.039 \
     --mag 14.15 \
     --M_over_L 3.0 \
     --M_sun 4.08 \
     --Re_arcsec 1.78

2) Update CSV in-place:
   python scripts/estimate_lens_mstar_re.py \
     --lens-id RXJ1131-1231 \
     --z_l 0.295 \
     --mag 17.88 \
     --M_over_L 3.0 \
     --M_sun 4.08 \
     --Re_arcsec 3.80 \
     --csv docs/lensing_targets.csv \
     --apply

Notes
- Magnitude system should match the solar absolute magnitude you supply in --M_sun.
  Defaults are for Cousins I-band (M_sun,I ≈ 4.08). Adjust if using a different band.
- This is a rough estimate: no K-correction, no extinction correction applied.
- Re conversion uses angular-diameter distance D_A(z) from astropy.cosmology.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
try:
    from astropy.cosmology import FlatLambdaCDM
except Exception as e:
    raise SystemExit("astropy is required. Try: pip install astropy")

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
ARCSEC_TO_RAD = 1.0 / 206265.0


def estimate_log10_mstar(app_mag: float, z_l: float, M_over_L: float = 3.0, M_sun: float = 4.08) -> float:
    # Luminosity distance in parsec
    DL_pc = COSMO.luminosity_distance(z_l).to('pc').value
    DM = 5.0 * np.log10(DL_pc / 10.0)
    M_abs = float(app_mag) - DM  # no K-corr/extinction
    # L/Lsun = 10^(-0.4 (M - M_sun))
    L_Lsun = 10 ** (-0.4 * (M_abs - float(M_sun)))
    M_star = float(M_over_L) * L_Lsun
    return float(np.log10(M_star))


def estimate_re_kpc(Re_arcsec: float, z_l: float) -> float:
    D_A_kpc = COSMO.angular_diameter_distance(z_l).to('kpc').value
    theta_rad = float(Re_arcsec) * ARCSEC_TO_RAD
    return float(D_A_kpc * theta_rad)


def update_csv(csv_path: Path, lens_id: str, log10M: Optional[float], Re_kpc: Optional[float], force: bool = False) -> None:
    # Read all rows
    with csv_path.open('r', encoding='utf-8') as f:
        r = list(csv.reader(f))
    if not r:
        raise SystemExit(f"Empty CSV: {csv_path}")
    header = r[0]
    rows = r[1:]
    # Column map
    col = {name: i for i, name in enumerate(header)}
    for field in ('lens_id', 'log10M_star', 'Re_kpc'):
        if field not in col:
            raise SystemExit(f"Missing required column '{field}' in {csv_path}")

    # Update matching lens_id (first match)
    updated = False
    for row in rows:
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        if row[col['lens_id']].strip() == lens_id:
            if log10M is not None:
                if force or row[col['log10M_star']].strip() == '':
                    row[col['log10M_star']] = f"{log10M:.3f}"
            if Re_kpc is not None:
                if force or row[col['Re_kpc']].strip() == '':
                    row[col['Re_kpc']] = f"{Re_kpc:.2f}"
            updated = True
            break
    if not updated:
        raise SystemExit(f"lens_id '{lens_id}' not found in {csv_path}")

    # Write back
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Estimate log10M_star and Re_kpc for a lens and optionally update CSV')
    ap.add_argument('--lens-id', required=True)
    ap.add_argument('--z_l', type=float, required=True)
    ap.add_argument('--mag', type=float, required=True, help='Apparent magnitude of the lens (band must match M_sun)')
    ap.add_argument('--M_over_L', type=float, default=3.0, help='M/L in the same band (solar units)')
    ap.add_argument('--M_sun', type=float, default=4.08, help='Solar absolute magnitude in the same band (e.g., I-band ≈ 4.08)')
    ap.add_argument('--Re_arcsec', type=float, default=None, help='Half-light radius in arcsec (if available)')
    ap.add_argument('--csv', type=str, default='', help='CSV path to update (e.g., docs/lensing_targets.csv)')
    ap.add_argument('--apply', action='store_true', help='Apply updates to the CSV (requires --csv)')
    ap.add_argument('--force', action='store_true', help='Overwrite existing CSV values')
    args = ap.parse_args()

    log10M = estimate_log10_mstar(float(args.mag), float(args.z_l), float(args.M_over_L), float(args.M_sun))
    Re_kpc = None
    if args.Re_arcsec is not None:
        Re_kpc = estimate_re_kpc(float(args.Re_arcsec), float(args.z_l))

    print(f"lens_id={args.lens_id}")
    print(f"  log10M_star ≈ {log10M:.3f}")
    if Re_kpc is not None:
        print(f"  Re_kpc ≈ {Re_kpc:.2f}")

    if args.apply:
        if not args.csv:
            raise SystemExit('--apply requires --csv')
        csv_path = Path(args.csv)
        update_csv(csv_path, args.lens_id, log10M, Re_kpc, force=bool(args.force))
        print(f"Updated {csv_path} for lens_id={args.lens_id}")


if __name__ == '__main__':
    main()

