#!/usr/bin/env python3
"""
analysis_alternative_rar_plateau.py — Demo analysis for the SVT-inspired RAR plateau alternative.

Outputs (under tariff/images/alternatives and tariff/results/alternatives):
- RAR g_obs vs g_bar scatter with plateau visible
- Rotation curve example showing added plateau component
- JSON summary of parameters and simple diagnostics

This script does NOT touch the main pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from .alternatives.rar_plateau_alt import PlateauParams, build_rar_curve
except Exception:
    from alternatives.rar_plateau_alt import PlateauParams, build_rar_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / 'tariff' / 'images' / 'alternatives'
RES_DIR = REPO_ROOT / 'tariff' / 'results' / 'alternatives'
IMG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

C_KM_S = 299_792.458


def run_demo(M_baryon_Msun: float = 5e10, p: PlateauParams = PlateauParams()):
    r_kpc = np.logspace(-1, 2.5, 400)
    g_bar, g_obs = build_rar_curve(r_kpc, M_baryon_Msun, p)

    # RAR diagram
    plt.figure(figsize=(8,6))
    plt.loglog(g_bar, g_obs, '.', alpha=0.7, label='RAR (alt plateau)')
    gline = np.logspace(-14, -8, 200)
    plt.loglog(gline, gline, '--', color='k', lw=1, label='g_obs = g_bar')
    plt.xlabel('g_bar [m s$^{-2}$]')
    plt.ylabel('g_obs [m s$^{-2}$]')
    plt.title('RAR with plateau (alternative)')
    plt.grid(alpha=0.3, which='both')
    plt.legend()
    rar_png = IMG_DIR / 'rar_plateau_alternative.png'
    plt.tight_layout(); plt.savefig(rar_png, dpi=150); plt.close()

    # Rotation curve illustration: v^2/r = g_obs → v = sqrt(g_obs * r)
    r_m = r_kpc * 3.08567758e19
    v_kms = np.sqrt(np.clip(g_obs * r_m, 0, np.inf)) / 1000.0
    vbar_kms = np.sqrt(np.clip(g_bar * r_m, 0, np.inf)) / 1000.0
    plt.figure(figsize=(8,6))
    plt.loglog(r_kpc, vbar_kms, '--', label='v_bar(r)')
    plt.loglog(r_kpc, v_kms, '-', label='v_obs(r) with plateau')
    plt.xlabel('r [kpc]'); plt.ylabel('v [km/s]')
    plt.title('Rotation curve with plateau component (alternative)')
    plt.grid(alpha=0.3, which='both'); plt.legend()
    rc_png = IMG_DIR / 'rotation_curve_plateau_alternative.png'
    plt.tight_layout(); plt.savefig(rc_png, dpi=150); plt.close()

    summary = {
        'M_baryon_Msun': float(M_baryon_Msun),
        'params': {'a_p': p.a_p, 'g_t': p.g_t, 'm': p.m},
        'figures': {
            'rar': str(rar_png),
            'rotation_curve': str(rc_png),
        },
        'notes': 'Toy SVT-inspired plateau model; main pipeline unchanged.'
    }
    out_json = RES_DIR / 'alternative_rar_plateau_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {rar_png}\nSaved {rc_png}\nWrote {out_json}')


if __name__ == '__main__':
    run_demo()
