#!/usr/bin/env python3
"""
Compute the Milky Way vertical force K_z(R0, z) and infer Σ_1.1 from the same
baryonic mass model used in the MW rotation figure (Miyamoto–Nagai thin, thick,
gas disks + Hernquist bulge). Outputs a CSV and a PNG for use in the paper.

Notes
- This is an initial baryons-only implementation (GR). An extension to include
  the RAR-plateau "phantom" component will require a consistent 3D mapping
  from xi(R) to an effective density ρ_eff(R,z); we keep that as a follow-up.
- We use the classic approximation K_z(z) ≈ 2π G Σ(<|z|) near the solar radius
  to infer Σ_1.1 from K_z(1.1 kpc). More precise treatments include the radial
  derivative term in the Poisson equation; we document that in README and plan
  a full treatment in the follow-up.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Constants
G_KPC = 4.300917270e-6  # kpc (km/s)^2 / Msun
TWOPI = 2.0 * np.pi


def default_baryon_params() -> dict:
    return {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }


def _mn_phi(R: np.ndarray, z: np.ndarray, M: float, a: float, b: float) -> np.ndarray:
    B = np.sqrt(z*z + b*b)
    return -G_KPC * M / np.sqrt(R*R + (a + B)**2)


def _mn_kz(R: np.ndarray, z: np.ndarray, M: float, a: float, b: float) -> np.ndarray:
    # Kz = -∂Φ/∂z
    # For MN: Φ = -GM / sqrt(R^2 + (a + B)^2), B = sqrt(z^2 + b^2)
    # ∂Φ/∂z = -GM * (a + B) * (z/B) / [ (R^2 + (a + B)^2)^(3/2) ]
    B = np.sqrt(z*z + b*b)
    denom = (R*R + (a + B)**2) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        dphi_dz = -G_KPC * M * (a + B) * (z / np.maximum(B, 1e-12)) / np.maximum(denom, 1e-30)
    return -dphi_dz  # Kz = -∂Φ/∂z


def _hern_phi(R: np.ndarray, z: np.ndarray, M: float, a: float) -> np.ndarray:
    r = np.sqrt(R*R + z*z)
    return -G_KPC * M / (r + a)


def _hern_kz(R: np.ndarray, z: np.ndarray, M: float, a: float) -> np.ndarray:
    r = np.sqrt(R*R + z*z)
    with np.errstate(divide='ignore', invalid='ignore'):
        dphi_dz = -G_KPC * M * z / (np.maximum(r, 1e-12) * (r + a)**2)
    return -dphi_dz


def compute_kz_components(R0_kpc: float, z_grid: np.ndarray, p: dict) -> dict:
    R = np.full_like(z_grid, float(R0_kpc))
    kz = {}
    if p.get('include_disk_thin', True):
        kz['thin'] = _mn_kz(R, z_grid, p['M_disk_thin_solar'], p['R_d_thin_kpc'], p['h_z_thin_kpc'])
    if p.get('include_disk_thick', True):
        kz['thick'] = _mn_kz(R, z_grid, p['M_disk_thick_solar'], p['R_d_thick_kpc'], p['h_z_thick_kpc'])
    if p.get('include_gas', True):
        kz['gas'] = _mn_kz(R, z_grid, p['M_gas_solar'], p['R_d_gas_kpc'], p['h_z_gas_kpc'])
    if p.get('include_bulge', True):
        kz['bulge'] = _hern_kz(R, z_grid, p['M_bulge_solar'], p['a_bulge_kpc'])
    return kz


def infer_sigma_from_kz(kz: np.ndarray, z_kpc: np.ndarray) -> np.ndarray:
    # Oort limit approximation: Kz(z) ≈ 2π G Σ(<|z|)
    # Return Σ_inferred(z) in Msun/kpc^2
    return np.maximum(kz, 0.0) / (TWOPI * G_KPC)


def main():
    ap = argparse.ArgumentParser(description='Compute MW Kz(R0,z) and Σ_1.1 from baryonic model (MN+Hernquist).')
    ap.add_argument('--R0-kpc', type=float, default=8.2, help='Solar radius (kpc)')
    ap.add_argument('--zmax-kpc', type=float, default=3.0, help='Max height (kpc)')
    ap.add_argument('--nz', type=int, default=181, help='Number of z samples (including 0)')
    ap.add_argument('--out-csv', type=str, default='results/mw_kz_sigma.csv')
    ap.add_argument('--out-png', type=str, default='images/mw_kz_sigma.png')
    args = ap.parse_args()

    p = default_baryon_params()
    z = np.linspace(0.0, float(args.zmax_kpc), int(args.nz))
    kz_comp = compute_kz_components(float(args.R0_kpc), z, p)
    kz_tot = np.zeros_like(z)
    for v in kz_comp.values():
        kz_tot += v

    sigma_inferred = infer_sigma_from_kz(kz_tot, z)
    # Σ_1.1 estimate
    z11 = 1.1
    sig11 = float(np.interp(z11, z, sigma_inferred))

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8') as f:
        cols = ['z_kpc', 'Kz_tot_km2s2_per_kpc'] + [f'Kz_{k}_km2s2_per_kpc' for k in kz_comp.keys()] + ['Sigma_inferred_Msun_per_kpc2']
        f.write(','.join(cols) + '\n')
        for i in range(len(z)):
            row = [f'{z[i]:.6f}', f'{kz_tot[i]:.6e}'] + [f'{kz_comp[k][i]:.6e}' for k in kz_comp.keys()] + [f'{sigma_inferred[i]:.6e}']
            f.write(','.join(row) + '\n')

    # Plot
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(z, kz_tot, 'k-', lw=2, label='Kz total (baryons)')
    for k, v in kz_comp.items():
        plt.plot(z, v, lw=1.5, label=f'Kz {k}')
    plt.axvline(z11, color='tab:orange', ls='--', label=f'z=1.1 kpc → Σ~{sig11/1e6:.2f}×10^6 Msun/kpc^2')
    plt.xlabel('z (kpc)'); plt.ylabel('Kz (km^2 s^-2 kpc^-1)')
    plt.title(f'MW vertical force at R0={args.R0_kpc} kpc (baryons only)')
    plt.grid(alpha=0.3); plt.legend(frameon=False)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

    print(f'Wrote {out_csv} and {out_png}. Σ_1.1 ≈ {sig11:.3e} Msun/kpc^2 (Oort approximation).')


if __name__ == '__main__':
    main()
