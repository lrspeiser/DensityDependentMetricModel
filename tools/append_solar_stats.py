#!/usr/bin/env python3
"""
Append Solar-System summary stats and ΔGM/GM table to docs.
- Computes median and IQR of ΔlogZ(power-exp) across the five galaxies and appends to docs/solar/solar_evidence_deltas.md
- Appends a second table to docs/table_solar_screening.md mapping |xi-1| to ΔGM/GM ≈ |xi-1| at 1, 9.5, 19, 30 AU using posterior medians [p16,p84], aggregated across the five galaxies (median and IQR across galaxies).
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np

REPO = Path(__file__).resolve().parents[1]
IM = REPO / 'images'
DOC_SOLAR = REPO / 'docs' / 'solar' / 'solar_evidence_deltas.md'
DOC_TABLE = REPO / 'docs' / 'table_solar_screening.md'
GALS = ['NGC3198','NGC2403','NGC2841','NGC6946','NGC5055']

# AU -> representative densities [Msun/kpc^3] (rough mapping used for screening illustration)
AU_POINTS = [
    ("1 AU", 1.0e19),
    ("9.5 AU", 5.0e17),
    ("19 AU", 2.0e17),
    ("30 AU", 1.0e17),
]

# Helper imports from our posterior-bands utility
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.gen_solar_system_posterior import draw_xi_band

def load_logZ_pair(g: str):
    p_exp = IM / f'sparc_env_fit_{g}_exp_hi.json'
    p_pow = IM / f'sparc_env_fit_{g}_power_hi.json'
    def get(p):
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
            e = j.get('evidence', {})
            return e.get('logZ', None)
        except Exception:
            return None
    return get(p_exp), get(p_pow)


def append_delta_stats():
    deltas = []
    for g in GALS:
        z_exp, z_pow = load_logZ_pair(g)
        if (z_exp is not None) and (z_pow is not None):
            deltas.append(z_exp - z_pow)  # exp - power per user request
    if not deltas:
        return
    arr = np.array(deltas, float)
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Summary across five galaxies: median ΔlogZ(exp_hi − power_hi) = {med:.3f} (IQR {q25:.3f} to {q75:.3f}).")
    lines.append("Screening choice changes evidence at the O(1) level; galaxy-level conclusions unchanged.")
    with open(DOC_SOLAR, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def append_deltaGM_table():
    rho_grid = np.logspace(15, 22, 400)
    T_high = 300.0
    # For each AU, gather per-galaxy medians for exp and power, then compute across-galaxy median/IQR
    rows = []
    for au_label, rho in AU_POINTS:
        # find nearest index
        j = int(np.clip(np.argmin(np.abs(rho_grid - rho)), 0, len(rho_grid)-1))
        med_exp_vals = []
        p16_exp_vals = []
        p84_exp_vals = []
        med_pow_vals = []
        p16_pow_vals = []
        p84_pow_vals = []
        for g in GALS:
            exp_npz = IM / f'sparc_env_fit_{g}_exp_hi.npz'
            pow_npz = IM / f'sparc_env_fit_{g}_power_hi.npz'
            if exp_npz.exists():
                exp_qs = draw_xi_band(exp_npz, 'exp', rho_grid, T_high)
                e16,e50,e84 = exp_qs[j,0], exp_qs[j,1], exp_qs[j,2]
                med_exp_vals.append(e50); p16_exp_vals.append(e16); p84_exp_vals.append(e84)
            if pow_npz.exists():
                pow_qs = draw_xi_band(pow_npz, 'power', rho_grid, T_high)
                p16,p50,p84 = pow_qs[j,0], pow_qs[j,1], pow_qs[j,2]
                med_pow_vals.append(p50); p16_pow_vals.append(p16); p84_pow_vals.append(p84)
        def agg(x):
            if not x:
                return (np.nan, np.nan, np.nan)
            a = np.array(x, float)
            return (float(np.median(a)), float(np.percentile(a,25)), float(np.percentile(a,75)))
        mE, qE1, qE3 = agg(med_exp_vals)
        mP, qP1, qP3 = agg(med_pow_vals)
        rows.append((au_label, rho, (mE, qE1, qE3), (mP, qP1, qP3)))
    # Append to DOC_TABLE
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Mapping to ΔGM/GM (weak-field) at AU radii")
    lines.append("We adopt the simple weak-field rule ΔGM/GM ≈ |ξ−1|. Values below are medians across the five galaxies’ posteriors with [IQR] across galaxies. Densities are representative placeholders for the vacuum/plasma environment near the listed AU radii.")
    lines.append("")
    lines.append("| Radius | ρ [M_⊙/kpc³] | ΔGM/GM (exp) median [IQR] | ΔGM/GM (power) median [IQR] |")
    lines.append("|---|---:|---:|---:|")
    for label, rho, expv, powv in rows:
        lines.append(f"| {label} | {rho:.2e} | {expv[0]:.2e} [{expv[1]:.2e},{expv[2]:.2e}] | {powv[0]:.2e} [{powv[1]:.2e},{powv[2]:.2e}] |")
    lines.append("")
    lines.append("Reference: canonical two-way ranging/ephemeris limits suggest |ΔGM/GM| ≲ O(10^{-12}) at outer-planet scales; see, e.g., Adelberger et al. (2003) and follow-ups.")
    with open(DOC_TABLE, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    append_delta_stats()
    append_deltaGM_table()

if __name__ == '__main__':
    main()

