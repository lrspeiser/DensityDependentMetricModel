#!/usr/bin/env python3
# Sandbox MW Kz variant: compute baryons-only Kz and record gate config for DG phantom
# experiments, without modifying the main module.

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from xi_registry_variants import build_gate

G_CGS = 6.67430e-8


def main():
    ap = argparse.ArgumentParser(description='MW Kz sandbox (baryons-only + gate config)')
    ap.add_argument('--gate', default='density', choices=['accel','density','hybrid'])
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Baryons-only placeholder (for illustration). Real Kz would integrate the disk/bulge model.
    z_kpc = np.linspace(0.1, 2.0, 20)
    Kz_bary = 2.0 * np.pi * G_CGS * (55.0 * 1.989e33 / (3.086e21)**2) * (z_kpc/1.0) * 0.0  # dummy zeros

    (outdir/'mw_kz_baryons_only.csv').write_text(
        'z_kpc,Kz_cgs\n' + '\n'.join(f"{z:.3f},{k:.6e}" for z,k in zip(z_kpc, Kz_bary)), encoding='utf-8')

    gate_meta = {
        'gate': args.gate,
        'params': {'a0': args.a0, 'rho_c': args.rho_c, 'gamma': args.gamma, 'zeta': args.zeta, 'Dmax': args.Dmax},
        'note': 'DG phantom not computed here; this sandbox only records gate config.'
    }
    (outdir/'mw_kz_gate_meta.json').write_text(json.dumps(gate_meta, indent=2), encoding='utf-8')
    print('Wrote baryons-only Kz placeholder and gate meta to', outdir)


if __name__ == '__main__':
    main()
