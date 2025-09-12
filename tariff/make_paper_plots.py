#!/usr/bin/env python3
"""
make_paper_plots.py — tariff-only orchestrator to generate all paper figures and embed links.

Runs:
- Baselines: Hubble, CMB spectrum (if CSV present), Tolman (if CSV present), SN time-dilation (if CSV present)
- Unified gate: μ(z) overlay + χ², H_eff & BAO overlays (+ r_d fit if CSV present), JSON metrics

Writes images to tariff/images/ and JSON to tariff/results/.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, 'images')
RES = os.path.join(HERE, 'results')
os.makedirs(IMG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Robust imports when called as a script
try:
    from .analysis_baselines import baseline_hubble, fit_cmb_temperature, fit_tolman_exponent, fit_sntd_exponent
    from .analysis_unified_gate import analyze_unified_gate
    from .unified_gate_scaffold import GateParams, calibrate_kappa_to_cmb
except Exception:
    from analysis_baselines import baseline_hubble, fit_cmb_temperature, fit_tolman_exponent, fit_sntd_exponent
    from analysis_unified_gate import analyze_unified_gate
    from unified_gate_scaffold import GateParams, calibrate_kappa_to_cmb

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PANTHEON = str(REPO_ROOT / 'external_data' / 'pantheon' / 'Pantheon+SH0ES.dat')
CMB_CSV = str(REPO_ROOT / 'tariff' / 'data' / 'cmb_firas_like.csv')
TOLMAN_CSV = str(REPO_ROOT / 'tariff' / 'data' / 'tolman_sb.csv')
SNTD_CSV = str(REPO_ROOT / 'tariff' / 'data' / 'sn_timedilation.csv')
BAO_CSV = str(REPO_ROOT / 'tariff' / 'data' / 'bao_compilation.csv')

def main():
    # Baselines (always run Hubble; optional others if files present)
    baseline_hubble(PANTHEON)
    if os.path.exists(CMB_CSV):
        fit_cmb_temperature(CMB_CSV)
    if os.path.exists(TOLMAN_CSV):
        fit_tolman_exponent(TOLMAN_CSV)
    if os.path.exists(SNTD_CSV):
        fit_sntd_exponent(SNTD_CSV)

    # Unified gate
    kappa_guess = calibrate_kappa_to_cmb(f_void=0.8, D_LSS_Mpc=14000.0, G_cap_minus1=1.0)
    params = GateParams(eta=3.0, p=1.5, q=1.0, rho_star_evcm3=0.26, kappa_per_Mpc=kappa_guess)
    bao_path = BAO_CSV if os.path.exists(BAO_CSV) else None
    analyze_unified_gate(PANTHEON, bao_path, params)
    print("All paper plots and metrics generated under tariff/images/ and tariff/results/")

if __name__ == '__main__':
    sys.exit(main())
