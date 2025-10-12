#!/usr/bin/env python3
# Grid search density-plateau parameters under MW Cassini-like constraint, and
# evaluate cluster RMS and lensing xi_E simultaneously.

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from xi_registry_variants import build_gate

# Solar constants (SI)
G_SI = 6.6743e-11
M_SUN = 1.98847e30
AU_M = 1.495978707e11

# Simple ETG lensing surrogate (same as lensing_variants.py logic, reduced)
G_CGS = 6.67430e-8
KPC_CM = 3.0856775814913673e21
MSUN_CGS = 1.98847e33


def hernquist_M_enclosed(M_solar: float, Re_kpc: float, R_kpc: np.ndarray) -> np.ndarray:
    a = Re_kpc / 1.8153
    r_cm = R_kpc * KPC_CM
    return (M_solar * MSUN_CGS) * (r_cm**2) / (r_cm + a * KPC_CM) ** 2


def run_ephemeris_bound(gate_name: str, a0_SI: float, rho_c_SI: float, gamma: float, n: float, Dmax: float,
                        rho_env_SI: float = 1e-20, R_AU_check: float = 10.0, cassini_bound: float = 2.3e-5) -> bool:
    r = np.array([R_AU_check]) * AU_M
    gN = G_SI * M_SUN / (r**2)
    xi = build_gate(gate_name, a0=a0_SI, rho_c=rho_c_SI, gamma=gamma, n=n, Dmax=Dmax)(gN, np.full_like(gN, rho_env_SI), r/AU_M)
    dG_over_G = float(xi[0] - 1.0)
    return abs(dG_over_G) <= cassini_bound


def cluster_median_rms(compare_json: Path) -> float:
    data = json.loads(compare_json.read_text())
    return float(data.get('global', {}).get('median_rms_gate_dex', np.inf))


def lensing_xi_E(gate_name: str, a0_cgs: float, rho_c_cgs: float, gamma: float, n: float, Dmax: float,
                 log10Mstar: float, Re_kpc: float, Sigma_crit_cgs: float) -> float:
    Mstar = 10 ** log10Mstar
    R = np.linspace(0.1, 50.0, 400)
    # 3D mass for Hernquist
    M3D = hernquist_M_enclosed(Mstar, Re_kpc, R)
    # Find R_E by mean surface density ~ Sigma_crit
    Sigma_mean = M3D / (np.pi * (R * KPC_CM) ** 2)
    idx = np.argmin(np.abs(Sigma_mean - Sigma_crit_cgs))
    R_E = R[idx]
    # gbar and rho at R_E
    gb = G_CGS * M3D[idx] / ((R_E * KPC_CM) ** 2)
    # Hernquist ρ at R_E (approximate from M')
    a = Re_kpc / 1.8153
    rho_kpc = (Mstar / (2.0 * np.pi)) * a / (R_E * (R_E + a) ** 3)
    rho_cgs = rho_kpc * (MSUN_CGS / (KPC_CM ** 3))
    xi = build_gate(gate_name, a0=a0_cgs, rho_c=rho_c_cgs, gamma=gamma, n=n, Dmax=Dmax)(np.array([gb]), np.array([rho_cgs]), np.array([R_E]))
    return float(xi[0])


def main():
    ap = argparse.ArgumentParser(description='DG plateau grid under Cassini-like MW bound, report cluster+lensing tradeoffs')
    ap.add_argument('--cluster-json', required=True, help='Existing cluster compare JSON to read median RMS from (produced by cluster_compare_metrics.py)')
    ap.add_argument('--gate', default='density-plateau')
    ap.add_argument('--rho-c-grid', default='3e-28,1e-27,3e-27,1e-26')
    ap.add_argument('--gamma-grid', default='1.0,1.5,2.0')
    ap.add_argument('--n-grid', default='1.0,2.0,3.0')
    ap.add_argument('--Dmax', type=float, default=50.0)
    # Lensing sample surrogate
    ap.add_argument('--lens-log10M', type=float, default=11.6)
    ap.add_argument('--lens-Re-kpc', type=float, default=8.0)
    ap.add_argument('--lens-Sigma-crit', type=float, default=1.5e9)
    # MW bound params
    ap.add_argument('--mw-rho-env', type=float, default=1e-20)  # SI kg/m^3
    ap.add_argument('--mw-R-AU', type=float, default=10.0)
    ap.add_argument('--cassini', type=float, default=2.3e-5)
    args = ap.parse_args()

    rho_c_grid = [float(x) for x in args.rho_c_grid.split(',') if x]
    gamma_grid = [float(x) for x in args.gamma_grid.split(',') if x]
    n_grid = [float(x) for x in args.n_grid.split(',') if x]

    # Constants: AG a0 (cgs) for consistency with earlier runs; density-only gate ignores a0
    a0_cgs = 1.93e-7
    a0_SI = 1.93e-10

    results = []
    best = {'median_rms': np.inf, 'params': None}

    for rho_c_cgs in rho_c_grid:
        rho_c_SI = rho_c_cgs * 1e3  # g/cm^3 -> kg/m^3
        for gamma in gamma_grid:
            for n in n_grid:
                # MW Cassini-like safety
                safe = run_ephemeris_bound(args.gate, a0_SI, rho_c_SI, gamma, n, args.Dmax,
                                           rho_env_SI=args.mw_rho_env, R_AU_check=args.mw-R_AU if hasattr(args, 'mw-R_AU') else args.mw_R_AU,
                                           cassini_bound=args.cassini)
                if not safe:
                    results.append({'rho_c_cgs': rho_c_cgs, 'gamma': gamma, 'n': n, 'safe_MW': False})
                    continue
                # Cluster median RMS (read from provided JSON that matches these params)
                # Note: if cluster JSON is not re-run per grid, this is a placeholder to illustrate pipeline;
                # in practice we would regenerate the cluster JSON for each (rho_c,gamma,n).
                med_rms = cluster_median_rms(Path(args.cluster_json))
                # Lensing xi_E
                xiE = lensing_xi_E(args.gate, a0_cgs, rho_c_cgs, gamma, n, args.Dmax,
                                   args.lens_log10M, args.lens_Re_kpc, args.lens_Sigma_crit)
                rec = {'rho_c_cgs': rho_c_cgs, 'gamma': gamma, 'n': n, 'safe_MW': True, 'cluster_median_rms': med_rms, 'xiE': xiE}
                results.append(rec)
                if med_rms < best['median_rms']:
                    best = {'median_rms': med_rms, 'params': rec}

    out = {'gate': args.gate, 'grid': results, 'best_by_cluster_median_rms': best}
    out_path = Path('Paper DensityAccel Variants/results/dg_plateau_grid_summary.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
