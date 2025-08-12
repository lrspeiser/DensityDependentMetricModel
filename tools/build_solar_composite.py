#!/usr/bin/env python3
"""
Build a composite Solar-System constraints figure.
- Loads per-galaxy exp_hi and power_hi NPZs
- Plots |xi-1| vs density on log-log axes with fixed y-range [1e-14,1e-3]
- Overlays Cassini, LLR, and inverse-square bands
- Adds a small inset showing sensitivity to T_high (+/-1 dex) for one representative galaxy

Output: paper_assets/solar/solar_system_constraints_composite.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.gen_solar_system_posterior import draw_xi_band

GALS = ['NGC3198','NGC2403','NGC2841','NGC6946','NGC5055']
COLORS = {
    'NGC3198':'#1f77b4',
    'NGC2403':'#ff7f0e',
    'NGC2841':'#2ca02c',
    'NGC6946':'#d62728',
    'NGC5055':'#9467bd',
}

IM = REPO / 'images'
OUT = REPO / 'paper_assets' / 'solar' / 'solar_system_constraints_composite.png'


def main():
    rho_grid = np.logspace(15, 22, 300)
    T_high = 300.0
    fig, ax = plt.subplots(figsize=(9,6))

    for g in GALS:
        exp_npz = IM / f'sparc_env_fit_{g}_exp_hi.npz'
        pow_npz = IM / f'sparc_env_fit_{g}_power_hi.npz'
        exp_qs = draw_xi_band(exp_npz, 'exp', rho_grid, T_high)
        pow_qs = draw_xi_band(pow_npz, 'power', rho_grid, T_high)
        c = COLORS[g]
        ax.fill_between(rho_grid, exp_qs[:,0], exp_qs[:,2], color=c, alpha=0.18)
        ax.plot(rho_grid, exp_qs[:,1], color=c, lw=2, label=f'{g} exp')
        ax.fill_between(rho_grid, pow_qs[:,0], pow_qs[:,2], color=c, alpha=0.10)
        ax.plot(rho_grid, pow_qs[:,1], color=c, lw=2, ls='--', label=f'{g} power')

    # Constraint bands
    cassini = 2.3e-5
    invsq = 1e-12
    ax.axhspan(0, cassini, color='gray', alpha=0.15, label='Cassini |γ−1|<2.3×10⁻⁵')
    ax.axhspan(0, invsq, color='green', alpha=0.08, label='Inverse-square ~1×10⁻¹²')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e-3)
    ax.set_xlabel('Density ρ [M⊙/kpc³] (Solar-System range)')
    ax.set_ylabel('|ξ−1|')
    ax.set_title('Solar-System constraints from galaxy posteriors (exp solid, power dashed)')
    ax.legend(ncol=2, frameon=False, fontsize=9)

    # Sensitivity inset (use NGC3198 as representative)
    g = 'NGC3198'
    exp_npz = IM / f'sparc_env_fit_{g}_exp_hi.npz'
    pow_npz = IM / f'sparc_env_fit_{g}_power_hi.npz'
    axins = ax.inset_axes([0.62, 0.08, 0.35, 0.35])
    for shift, ls in [(-1.0, ':'), (0.0, '-'), (1.0, ':')]:
        Th = T_high * (10 ** shift)
        exp_qs = draw_xi_band(exp_npz, 'exp', rho_grid, Th)
        pow_qs = draw_xi_band(pow_npz, 'power', rho_grid, Th)
        axins.plot(rho_grid, exp_qs[:,1], color=COLORS[g], ls=ls, lw=1)
        axins.plot(rho_grid, pow_qs[:,1], color=COLORS[g], ls=ls, lw=1, alpha=0.7)
    axins.set_xscale('log'); axins.set_yscale('log')
    axins.set_ylim(1e-14, 1e-3)
    axins.set_title('T_high sensitivity (NGC3198)', fontsize=8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f'Saved {OUT}')

if __name__ == '__main__':
    main()

