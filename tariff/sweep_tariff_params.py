#!/usr/bin/env python3
"""
sweep_tariff_params.py — grid search over tariff knobs to match Pantheon+.

Confined to /tariff; imports energy_tariff_model via file path if needed.

Metrics:
- Chi^2_mu: χ^2 on distance modulus μ(z) using Pantheon+ (zHD, MU_SH0ES, MU_SH0ES_ERR_DIAG)
- Reduced χ^2 = χ^2 / (N-1)

Parameters swept (defaults; override via CLI):
- dmax_list: 30, 40, 50
- gbar_void_list: 1e-15, 3e-14, 1e-13, 1e-12 (m/s^2)
- r0_void_list: 0, 2000, 3000, 4000 (Mpc) — 0 disables f_void
- gamma_void_list: 1.0, 1.5, 2.0
- H0 anchor: 67.4 km/s/Mpc (used to calibrate k per combo)

Writes: tariff/sweep_results.csv (sorted by reduced χ^2 ascending)
Prints top 10 combos.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import List, Tuple

import numpy as np

# Attempt to import from file path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ETM_PATH = os.path.join(_THIS_DIR, 'energy_tariff_model.py')
if _ETM_PATH not in sys.modules:
    import importlib.util
    spec = importlib.util.spec_from_file_location('energy_tariff_model_runtime', _ETM_PATH)
    if spec and spec.loader:
        etm = importlib.util.module_from_spec(spec)
        sys.modules['energy_tariff_model_runtime'] = etm
        spec.loader.exec_module(etm)  # type: ignore[attr-defined]
    else:
        raise RuntimeError('Could not import energy_tariff_model.py for sweep.')
else:
    etm = sys.modules['energy_tariff_model_runtime']  # type: ignore[index]

# Shorthand
EnergyCouplingParams = getattr(etm, 'EnergyCouplingParams')
PhotonJourney = getattr(etm, 'PhotonJourney')
load_pantheon_data = getattr(etm, 'load_pantheon_data')
C_KM_S = getattr(etm, 'C_KM_S')


def chi2_mu(sim: PhotonJourney, z: np.ndarray, mu: np.ndarray, muerr: np.ndarray) -> Tuple[float, float, int]:
    """Compute χ^2 over μ(z). Returns (chi2, red_chi2, N_used)."""
    # Predict μ(z) using the sim's distance_modulus_at_z
    mu_model = sim.distance_modulus_at_z(z)
    mask = np.isfinite(mu_model) & np.isfinite(mu) & np.isfinite(muerr) & (muerr > 0)
    if not np.any(mask):
        return float('inf'), float('inf'), 0
    res = (mu[mask] - mu_model[mask]) / muerr[mask]
    chi2 = float(np.sum(res * res))
    dof = max(int(np.count_nonzero(mask) - 1), 1)
    return chi2, chi2 / dof, int(np.count_nonzero(mask))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Sweep tariff knobs to match Pantheon+')
    ap.add_argument('--data-file', type=str, default=os.path.join('external_data','pantheon','Pantheon+SH0ES.dat'))
    ap.add_argument('--out-csv', type=str, default=os.path.join('tariff','sweep_results.csv'))
    ap.add_argument('--h0', type=float, default=67.4, help='H0 anchor (km/s/Mpc) used to calibrate k')
    ap.add_argument('--dmax', type=str, default='30,40,50')
    ap.add_argument('--gbar-void', type=str, default='1e-15,3e-14,1e-13,1e-12')
    ap.add_argument('--r0-void', type=str, default='0,2000,3000,4000')
    ap.add_argument('--gamma-void', type=str, default='1.0,1.5,2.0')
    ap.add_argument('--energy-coupled', action='store_true')
    ap.add_argument('--zeta-energy', type=float, default=1.0)
    ap.add_argument('--beta-energy', type=float, default=2.0)
    ap.add_argument('--u-gamma-evcm3', type=float, default=0.26)
    ap.add_argument('--E0-evcm3', type=float, default=0.26)
    args = ap.parse_args(argv)

    # Load data
    z_data, mu_data, mu_err = load_pantheon_data(args.data_file)

    # Build grids
    def parse_list(s: str, conv=float):
        return [conv(x.strip()) for x in s.split(',') if x.strip()]
    dmax_list = parse_list(args.dmax, float)
    gbar_list = parse_list(args.gbar_void, float)
    r0_list = parse_list(args.r0_void, float)
    gamma_list = parse_list(args.gamma_void, float)

    # Energy coupling params
    eparams = EnergyCouplingParams(
        enabled=bool(args.energy_coupled),
        zeta_energy=float(args.zeta_energy),
        beta_energy=float(args.beta_energy),
        u_gamma_evcm3=float(args.u_gamma_evcm3),
        E0_evcm3=float(args.E0_evcm3),
    )

    rows = []
    count = 0
    total = len(dmax_list)*len(gbar_list)*len(r0_list)*len(gamma_list)
    for dmax in dmax_list:
        # Calibrate k from H0 and void cap (convention consistent with tariff model)
        k_val = float(args.h0) / (C_KM_S * (float(dmax) - 1.0)) if float(dmax) > 1.0 else 0.0
        for gbar in gbar_list:
            for r0 in r0_list:
                for gamma in gamma_list:
                    count += 1
                    sim = PhotonJourney(
                        k_coupling_mpc_inv=k_val,
                        energy_params=eparams,
                        d_max=float(dmax),
                        g_bar_void=float(gbar),
                        galaxy_shell_mpc=0.05,
                        r0_void=float(r0),
                        gamma_void=float(gamma),
                    )
                    chi2, red, n = chi2_mu(sim, z_data, mu_data, mu_err)
                    rows.append({
                        'dmax': dmax,
                        'gbar_void': gbar,
                        'r0_void': r0,
                        'gamma_void': gamma,
                        'k_mpc^-1': k_val,
                        'chi2': chi2,
                        'red_chi2': red,
                        'N_used': n,
                    })
                    if count % 10 == 0:
                        print(f"Progress {count}/{total} ... best red_chi2 so far: {min(r['red_chi2'] for r in rows):.3f}")

    # Sort by reduced chi2
    rows.sort(key=lambda r: r['red_chi2'])

    # Write CSV
    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote sweep results: {out_csv}")

    # Print top 10
    print("Top 10 by reduced χ^2:")
    for r in rows[:10]:
        print(f"  red_chi2={r['red_chi2']:.3f} | dmax={r['dmax']} gbar={r['gbar_void']:.2e} r0={r['r0_void']} gamma={r['gamma_void']} k={r['k_mpc^-1']:.3e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

