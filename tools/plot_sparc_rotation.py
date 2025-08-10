#!/usr/bin/env python3
"""
Plot TFR/GR predictions against a SPARC/Rotmod galaxy file.

Example (NGC 3198):
  python tools/plot_sparc_rotation.py \
    --file external_data/Rotmod_LTG/NGC3198_rotmod.dat \
    --name NGC 3198 \
    --er --lambda_max 4.0 --R0 15 --sigma_lnR 0.7 --w_min 0.02 --ups_disk 0.5 --ups_bul 0.7

M33:
  python tools/plot_sparc_rotation.py \
    --file external_data/Rotmod_LTG/NGC0598_rotmod.dat --name M33 --er

Outputs image under images/sparc_<name_slug>.png
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.sparc_loader import load_rotmod
from models.er_sparc import v_bar_from_components, v_er_from_components, xi_log_normal_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to SPARC/Rotmod .dat file")
    ap.add_argument("--name", default=None, help="Galaxy name for title/filename")
    ap.add_argument("--er", action="store_true", help="Overlay ER curve")
    ap.add_argument("--lambda_max", type=float, default=4.0)
    ap.add_argument("--R0", type=float, default=15.0, help="Band center [kpc]")
    ap.add_argument("--sigma_lnR", type=float, default=0.7)
    ap.add_argument("--w_min", type=float, default=0.02)
    ap.add_argument("--ups_disk", type=float, default=0.5, help="Stellar disk mass-to-light (3.6um) ")
    ap.add_argument("--ups_bul", type=float, default=0.7, help="Stellar bulge mass-to-light (3.6um)")
    ap.add_argument("--out", default=None, help="Optional output file path for the plot PNG")
    args = ap.parse_args()

    data = load_rotmod(args.file)
    R = data["R_kpc"]
    Vobs = data["Vobs_kms"]
    eV = data["eVobs_kms"]
    Vgas = data["Vgas_kms"]
    Vdisk = data["Vdisk_kms"]
    Vbul = data["Vbul_kms"]

    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, args.ups_disk, args.ups_bul)

    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or Path(args.file).stem.replace("_rotmod","")
    name_slug = name.lower().replace(" ", "_")

    plt.figure(figsize=(10,7))
    # Data
    plt.errorbar(R, Vobs, yerr=eV, fmt='o', color='k', ms=4, lw=1, alpha=0.8, label='Observed (SPARC)')
    # GR/baryon-only
    plt.plot(R, vbar, 'b--', lw=2, label='GR (baryons)')

    if args.er:
        vbar2, xi, ver = v_er_from_components(R, Vgas, Vdisk, Vbul, args.ups_disk, args.ups_bul,
                                              args.lambda_max, args.R0, args.sigma_lnR, args.w_min)
        # Split at max data R for extrapolation shading on a fine grid
        R_grid = np.linspace(max(1e-3, R.min()), max(R.max()*1.2, R.max()+5), 400)
        vbar_g = np.interp(R_grid, R, vbar)
        xi_g = xi_log_normal_R(R_grid, args.lambda_max, args.R0, args.sigma_lnR, args.w_min)
        ver_g = np.sqrt(np.clip(xi_g, 0, None)) * vbar_g
        R_data_max = float(np.max(R))
        m_in = R_grid <= R_data_max
        m_out = ~m_in
        if np.any(m_in):
            plt.plot(R_grid[m_in], ver_g[m_in], 'r-', lw=2.5, label='TFR — constrained')
        if np.any(m_out):
            plt.plot(R_grid[m_out], ver_g[m_out], color='#FF8C00', ls='--', lw=2.5, label='TFR — extrapolation')
            plt.axvspan(R_data_max, R_grid.max(), color='#FFA500', alpha=0.08)
        plt.axvline(R_data_max, color='k', ls=':', alpha=0.6, label=f"Max data R ≈ {R_data_max:.1f} kpc")
        print(f"xi_max = 1 + lambda_max = {1.0 + args.lambda_max:.3f}")

    plt.xlabel('R (kpc)')
    plt.ylabel('Vc (km s^{-1})')
    plt.title(f'{name}: SPARC vs GR vs TFR')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.xlim(0, max(R.max()*1.2, R.max()+5))
    ymax = max(np.nanmax(Vobs+eV), np.nanmax(vbar)*1.2)
    plt.ylim(0, max(300, float(ymax)+40))

    out_file = Path(args.out) if args.out else (out_dir / f"sparc_{name_slug}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
