#!/usr/bin/env python3
"""
analyze_wide_binaries.py

Predict wide-binary (WB) kinematic deviations under the RAR-plateau model in the
Solar neighborhood, focusing on the velocity ratio statistic vs projected
separation. This script produces a theory curve that can be compared with Gaia
DR3 WB analyses in the literature.

Notes
- No web access or API keys are used. If you want to overlay observed WB
  statistics, provide a vetted CSV via --wb-csv; otherwise, the script writes a
  pure-theory prediction for reproducibility.
- The prediction uses a simple two-body mapping: for a representative total
  stellar mass M ~ 2 M_sun, at separation r the Newtonian acceleration is
  g_N = G M / r^2. The RAR-plateau enhancement D = xi ≡ g_eff/g_bar yields
  v_model / v_Newton ≈ sqrt(D). We plot sqrt(D) − 1 vs r (AU) as a proxy for
  the fractional velocity excess expected under the model.

Usage
  python scripts/analyze_wide_binaries.py \
    --run-dir runs/rar_plateau_mw_full \
    --out-root results/next_steps/rar_plateau_mw_full \
    --images-root images/next_steps/rar_plateau_mw_full

Optional overlay from a local CSV (columns: s_AU, v_ratio)
  python scripts/analyze_wide_binaries.py \
    --run-dir runs/rar_plateau_mw_full \
    --out-root results/next_steps/rar_plateau_mw_full \
    --images-root images/next_steps/rar_plateau_mw_full \
    --wb-csv path/to/your_wb_summary.csv

"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

G_SI = 6.6743e-11
M_SUN = 1.98847e30
AU_M = 1.495978707e11

# Import xi function from orchestrator (pure NumPy path)
try:
    from scripts.next_steps_from_run import xi_rar_plateau_numpy  # type: ignore
except Exception:
    xi_rar_plateau_numpy = None  # type: ignore


def load_run_params(run_dir: Path) -> Dict[str, float]:
    # Prefer metadata snapshot from orchestrator; fall back to defaults
    md = Path('results') / 'next_steps' / run_dir.name / 'run_metadata.json'
    if md.exists():
        try:
            j = json.loads(md.read_text(encoding='utf-8'))
            rp = j.get('rar_plateau_params', {})
            if isinstance(rp, dict):
                return rp
        except Exception:
            pass
    # Fallback
    return {'a0_m_s2': 1.2e-10, 'zeta_env': 0.0}


def predict_vratio_curve(params: Dict[str, float], *, M_tot_Msun: float = 2.0,
                         s_min_AU: float = 1.0e3, s_max_AU: float = 2.0e5, n: int = 120) -> np.ndarray:
    """Return array with columns: s_AU, v_ratio_minus1 ( ≈ sqrt(xi) - 1 )."""
    s_AU = np.logspace(math.log10(s_min_AU), math.log10(s_max_AU), int(n))
    # Build a synthetic circular orbit mapping for xi evaluation
    # Use r in kpc and Vbar from g_N
    KPC_M = 3.085677581491367e19
    r_m = s_AU * AU_M
    r_kpc = r_m / KPC_M
    M_tot = float(M_tot_Msun) * M_SUN
    g_N = G_SI * M_tot / np.maximum(r_m**2, 1.0)
    V_ms = np.sqrt(np.maximum(g_N * r_m, 0.0))
    V_kms = V_ms / 1000.0
    if xi_rar_plateau_numpy is None:
        xi = np.ones_like(V_kms)
    else:
        xi, _ = xi_rar_plateau_numpy(
            V_kms, r_kpc,
            a0_m_s2=float(params.get('a0_m_s2', 1.2e-10)),
            zeta_env=float(params.get('zeta_env', 0.0)),
            rho=None, rho_c=params.get('rho_c', None),
            gamma_exp=float(params.get('gamma_exp', 3.0)),
            T0=params.get('T0', None), sigma_lnT=params.get('sigma_lnT', None),
            wmin=float(params.get('wmin', 0.0))
        )
    v_ratio_minus1 = np.sqrt(np.maximum(xi, 0.0)) - 1.0
    return np.vstack([s_AU, v_ratio_minus1]).T


def main():
    ap = argparse.ArgumentParser(description='Predict wide-binary statistic under RAR-plateau')
    ap.add_argument('--run-dir', required=True, help='Run directory name used to locate metadata (e.g., runs/rar_plateau_mw_full)')
    ap.add_argument('--out-root', required=True, help='Results root (e.g., results/next_steps/rar_plateau_mw_full)')
    ap.add_argument('--images-root', required=True, help='Images root (e.g., images/next_steps/rar_plateau_mw_full)')
    ap.add_argument('--wb-csv', default=None, help='Optional wide-binary CSV with columns: s_AU, v_ratio')
    ap.add_argument('--Mtot', type=float, default=2.0, help='Representative total stellar mass in M_sun for the theory curve')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_root)
    images_root = Path(args.images_root)
    out_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    params = load_run_params(run_dir)

    # Theory curve
    arr = predict_vratio_curve(params, M_tot_Msun=float(args.Mtot))
    out_csv = out_root / 'wide_binaries_pred.csv'
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('s_AU,v_ratio_minus1_theory\n')
        for s, vr in arr:
            f.write(f'{s:.6e},{vr:.6e}\n')

    # Optional overlay
    obs_s = None
    obs_v = None
    if args.wb_csv:
        p = Path(args.wb_csv)
        if p.exists():
            xs = []
            ys = []
            with p.open('r', encoding='utf-8') as f:
                header = [h.strip() for h in f.readline().strip().split(',')]
                cm = {h:i for i,h in enumerate(header)}
                for line in f:
                    if not line.strip():
                        continue
                    parts = [s.strip() for s in line.strip().split(',')]
                    try:
                        xs.append(float(parts[cm['s_AU']]))
                        ys.append(float(parts[cm['v_ratio']]))
                    except Exception:
                        continue
            if len(xs) >= 4:
                obs_s = np.asarray(xs, float)
                obs_v = np.asarray(ys, float)

    # Plot
    plt.figure(figsize=(6.8, 4.4))
    plt.semilogx(arr[:,0], arr[:,1], 'r-', lw=2, label='RAR-plateau theory (sqrt(ξ) − 1)')
    if obs_s is not None and obs_v is not None:
        plt.semilogx(obs_s, obs_v, 'k.', ms=4, alpha=0.5, label='Observed (WB summary)')
    plt.axhline(0.0, color='k', ls=':', lw=1)
    plt.xlabel('Projected separation s (AU)')
    plt.ylabel('Velocity ratio − 1')
    plt.title('Wide-binary kinematic prediction (Solar neighborhood)')
    plt.grid(alpha=0.3, which='both')
    plt.legend(frameon=False)
    figp = images_root / 'wide_binaries_pred.png'
    plt.tight_layout(); plt.savefig(figp, dpi=140); plt.close()

    print(f'Wrote {out_csv} and {figp}')


if __name__ == '__main__':
    main()

