#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster RAR pipeline (offline, reproducible)
- Total acceleration g_tot(r) from Umetsu+2016 NFW fits (CLASH) per cluster
- Baryonic acceleration g_bar(r) from ACCEPT electron-density shells
- Flexible theory curves (Newtonian, MOND-like, Emergent-gravity-like) and a0 fitting

Cosmology and mass definitions follow Umetsu et al. (2016, ApJ 821, 116):
Ω_m = 0.27, Ω_Λ = 0.73, H0 = 70 km/s/Mpc; M_200c & c_200c as in their Table 2.

Outputs (written under repo root if not overridden):
- results/cluster_rar/cluster_rar_points.csv (per-point table)
- results/cluster_rar/summary.json (cosmology, counts, best-fit a0 for two models)
- images/cluster_rar/cluster_rar_scatter.png (log10 g_bar vs log10 g_tot)

Notes:
- No internet I/O. Uses only external_data/accept_database.dat and hard-coded Umetsu+2016 parameters.
- Headless plotting (Agg backend) so it can run in CI.
"""

from __future__ import annotations
import os
import re
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# ----------------------------
# Physical constants (cgs)
# ----------------------------
G = 6.67430e-8               # cm^3 g^-1 s^-2
M_sun = 1.98847e33           # g
kpc = 3.085677581491367e21   # cm
Mpc = 3.085677581491367e24   # cm
m_p = 1.67262192369e-24      # g

# ----------------------------
# Cosmology (Umetsu+2016 defaults)
# ----------------------------
@dataclass(frozen=True)
class Cosmology:
    H0_km_s_Mpc: float = 70.0
    Omega_m: float = 0.27
    Omega_L: float = 0.73

    @property
    def H0_cgs(self) -> float:
        # H0 in s^-1
        return (self.H0_km_s_Mpc * 1.0e5) / Mpc

    def Ez(self, z: float) -> float:
        return math.sqrt(self.Omega_m * (1.0 + z) ** 3 + self.Omega_L)

    def rho_crit_cgs(self, z: float) -> float:
        # ρ_c(z) = 3 H(z)^2 / (8π G)
        Hz = self.H0_cgs * self.Ez(z)
        return 3.0 * Hz * Hz / (8.0 * math.pi * G)


COSMO = Cosmology()  # defaults match Umetsu+2016


# ----------------------------
# NFW halo (M200c, c200c) utilities
# ----------------------------
@dataclass
class NFW:
    M200c_Msun_h70: float   # in units of 1e14 Msun h70^-1 from Table 2
    c200c: float
    z: float
    cosmo: Cosmology = COSMO

    def __post_init__(self):
        # Convert M200c to grams (h70=1.0 for H0=70)
        self.M200c_g = self.M200c_Msun_h70 * 1.0e14 * M_sun  # Msun -> g
        # Critical density at z
        self.rho_c = self.cosmo.rho_crit_cgs(self.z)  # g cm^-3
        # r200c from M200c = (4/3)π 200 ρ_c r200c^3
        self.r200c_cm = ((3.0 * self.M200c_g) / (4.0 * math.pi * 200.0 * self.rho_c)) ** (1.0 / 3.0)
        # scale radius
        self.rs_cm = self.r200c_cm / self.c200c
        # delta_c and rho_s
        self.delta_c = (200.0 / 3.0) * (self.c200c ** 3) / (math.log(1.0 + self.c200c) - self.c200c / (1.0 + self.c200c))
        self.rho_s = self.delta_c * self.rho_c  # g cm^-3

    def M_enclosed(self, r_cm: np.ndarray) -> np.ndarray:
        """
        NFW 3D enclosed mass M(<r).
        """
        x = np.maximum(r_cm / self.rs_cm, 1e-12)
        term = np.log(1.0 + x) - (x / (1.0 + x))
        return 4.0 * math.pi * self.rho_s * (self.rs_cm ** 3) * term

    def g_tot(self, r_cm: np.ndarray) -> np.ndarray:
        """
        Total gravitational acceleration g=G M(<r)/r^2 (cgs).
        """
        Menc = self.M_enclosed(r_cm)
        return G * Menc / (r_cm ** 2)

    @property
    def r200c_kpc(self) -> float:
        return self.r200c_cm / kpc

    @property
    def rs_kpc(self) -> float:
        return self.rs_cm / kpc


# ----------------------------
# CLASH (Umetsu+2016) cluster parameters
#  - Redshift (Table 1)
#  - NFW (M200c, c200c) (Table 2)
# Units: M200c numbers are in 1e14 Msun h70^-1 (h70=1 here)
# ----------------------------
CLASH_PARAMS: Dict[str, Dict[str, float]] = {
    # X-ray-selected (Table 1 & Table 2)
    "Abell 383":              {"z": 0.187, "M200c": 7.98,  "c200c": 5.9},
    "Abell 209":              {"z": 0.206, "M200c": 15.40, "c200c": 2.7},
    "Abell 2261":             {"z": 0.224, "M200c": 23.10, "c200c": 3.7},
    "RX J2129.7+0005":       {"z": 0.234, "M200c": 6.14,  "c200c": 5.6},
    "Abell 611":              {"z": 0.288, "M200c": 15.76, "c200c": 3.9},
    "MS2137-2353":           {"z": 0.313, "M200c": 13.56, "c200c": 2.7},
    "RX J2248.7-4431":       {"z": 0.348, "M200c": 18.78, "c200c": 3.6},  # Abell S1063
    "MACS J1115.9+0129":     {"z": 0.352, "M200c": 16.66, "c200c": 3.0},
    "MACS J1931.8-2635":     {"z": 0.352, "M200c": 15.28, "c200c": 4.4},
    "RX J1532.9+3021":       {"z": 0.363, "M200c": 5.98,  "c200c": 5.2},
    "MACS J1720.3+3536":     {"z": 0.391, "M200c": 14.50, "c200c": 4.1},
    "MACS J0429.6-0253":     {"z": 0.399, "M200c": 9.76,  "c200c": 4.6},
    "MACS J1206.2-0847":     {"z": 0.440, "M200c": 18.17, "c200c": 3.7},
    "MACS J0329.7-0211":     {"z": 0.450, "M200c": 8.65,  "c200c": 6.7},
    "RX J1347.5-1145":       {"z": 0.451, "M200c": 34.25, "c200c": 3.2},
    "MACS J0744.9+3927":     {"z": 0.686, "M200c": 18.03, "c200c": 3.5},
    # High-magnification (Table 2)
    "MACS J0416.1-2403":     {"z": 0.396, "M200c": 10.74, "c200c": 2.9},
    "MACS J1149.5+2223":     {"z": 0.544, "M200c": 25.02, "c200c": 2.1},
    "MACS J0717.5+3745":     {"z": 0.548, "M200c": 26.77, "c200c": 1.8},
    "MACS J0647.7+7015":     {"z": 0.584, "M200c": 13.90, "c200c": 4.1},
}

# Name normalization to match ACCEPT naming idiosyncrasies
_def_rxj_aliases = {
    'abells1063': 'RX J2248.7-4431',
    'rxj13475-1145': 'RX J1347.5-1145',
}


def norm_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '', s).lower()


# ----------------------------
# ACCEPT parser and baryon mass builder
# ----------------------------
@dataclass
class AcceptShell:
    cluster: str
    Rin_Mpc: float
    Rout_Mpc: float
    ne_cm3: float
    ne_err_cm3: float
    T_keV: float  # optional


def parse_accept(path: str) -> Dict[str, List[AcceptShell]]:
    """
    Parse ACCEPT-like table into per-cluster shell lists.
    Expected header columns include:
      Name Rin Rout nelec neerr ... Tx Txerr ... (whitespace-separated)
    """
    clusters: Dict[str, List[AcceptShell]] = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#') or line.startswith('###'):
                continue
            parts = line.split()
            # Minimal safety: need at least Name, Rin, Rout, nelec, neerr
            try:
                name = parts[0]
                Rin = float(parts[1])
                Rout = float(parts[2])
                ne = float(parts[3])
                neerr = float(parts[4])
            except Exception:
                # skip malformed line
                continue
            # Attempt to read Tx (optional); tolerate missing
            Tx = float('nan')
            for idx in (13, 12, 11, 10):
                if idx < len(parts):
                    try:
                        Tx = float(parts[idx])
                        break
                    except Exception:
                        pass
            sh = AcceptShell(name, Rin, Rout, ne, neerr, Tx)
            clusters.setdefault(name, []).append(sh)

    # sort shells by radius
    for k in clusters:
        clusters[k].sort(key=lambda sh: sh.Rout_Mpc)
    return clusters


def build_gas_profiles(shells: List[AcceptShell], mu_e: float = 1.17) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From ACCEPT shells -> cumulative gas mass profile and radii.
    Returns:
      r_cm: radii at shell outer edges (cm)
      Mgas_cum_g: cumulative gas mass within r (g)
      g_bar_cgs: baryonic acceleration G*M_bary/r^2 (gas-only by default)
    """
    r_edges_cm = np.array([sh.Rout_Mpc * Mpc for sh in shells], dtype=float)
    Mgas_cum = []
    total = 0.0
    for sh in shells:
        rin = sh.Rin_Mpc * Mpc
        rout = sh.Rout_Mpc * Mpc
        vol = (4.0 * math.pi / 3.0) * (rout**3 - rin**3)
        rho_gas = mu_e * m_p * sh.ne_cm3  # g cm^-3
        dM = rho_gas * vol
        total += dM
        Mgas_cum.append(total)
    Mgas_cum_g = np.array(Mgas_cum, dtype=float)
    g_bar = G * Mgas_cum_g / (r_edges_cm ** 2)
    return r_edges_cm, Mgas_cum_g, g_bar


# ----------------------------
# Theory curves & a0 fitting
# ----------------------------

def g_pred_newton(g_bar: np.ndarray) -> np.ndarray:
    return g_bar


def g_pred_mond_simple(g_bar: np.ndarray, a0_cgs: float) -> np.ndarray:
    # "simple" interpolating function nu: g = 0.5 gN + sqrt(0.25 gN^2 + gN a0)
    return 0.5 * g_bar + np.sqrt(0.25 * g_bar**2 + g_bar * a0_cgs)


def g_pred_emergent(g_bar: np.ndarray, a0_cgs: float) -> np.ndarray:
    # heuristic EG-like: g = g_b + sqrt(a0 * g_b)
    return g_bar + np.sqrt(a0_cgs * g_bar)


def fit_a0(g_bar: np.ndarray,
           g_tot: np.ndarray,
           model: Callable[[np.ndarray, float], np.ndarray],
           a0_min: float = 1e-12, a0_max: float = 1e-7, n_grid: int = 4000) -> Tuple[float, float]:
    """
    Brute-force grid search for a0 (cgs) minimizing squared residuals in log-space.
    Returns best a0 and RMS scatter (dex).
    """
    mask = (g_bar > 0) & (g_tot > 0) & np.isfinite(g_bar) & np.isfinite(g_tot)
    if not np.any(mask):
        return float('nan'), float('nan')
    gb = g_bar[mask]
    gt = g_tot[mask]
    a0s = np.logspace(math.log10(a0_min), math.log10(a0_max), n_grid)
    best = None
    best_rms = 1e9
    for a0 in a0s:
        pred = model(gb, a0)
        resid = np.log10(gt) - np.log10(pred)
        rms = np.sqrt(np.mean(resid**2))
        if rms < best_rms:
            best_rms = rms
            best = a0
    return best if best is not None else float('nan'), best_rms


# ----------------------------
# Pipeline
# ----------------------------
@dataclass
class ClusterRARResult:
    cluster: str
    z: float
    radii_kpc: np.ndarray
    g_bar: np.ndarray
    g_tot: np.ndarray
    nfw: NFW


def _canonicalize_clash_key(name: str) -> Optional[str]:
    nm = norm_name(name)
    nm_map = {norm_name(k): k for k in CLASH_PARAMS.keys()}
    if nm in nm_map:
        return nm_map[nm]
    # try alias map
    if nm in _def_rxj_aliases:
        alias = _def_rxj_aliases[nm]
        nma = norm_name(alias)
        if nma in nm_map:
            return nm_map[nma]
    # try removing spaces/underscores and case-insensitive punctuation
    nm2 = norm_name(name.replace(' ', '').replace('_', ''))
    if nm2 in nm_map:
        return nm_map[nm2]
    return None


def run_cluster(cluster_name: str,
                accept_shells: List[AcceptShell],
                mu_e: float = 1.17) -> Optional[ClusterRARResult]:
    clash_name = _canonicalize_clash_key(cluster_name)
    if clash_name is None:
        return None
    pars = CLASH_PARAMS[clash_name]
    nfw = NFW(M200c_Msun_h70=pars["M200c"], c200c=pars["c200c"], z=pars["z"])

    r_cm, Mgas_cum_g, gbar = build_gas_profiles(accept_shells, mu_e=mu_e)
    gtot = nfw.g_tot(r_cm)
    return ClusterRARResult(cluster=clash_name,
                            z=pars["z"],
                            radii_kpc=r_cm / kpc,
                            g_bar=gbar,
                            g_tot=gtot,
                            nfw=nfw)


def run_all(accept_path: str,
            out_results: str,
            out_images: str,
            mu_e: float = 1.17,
            save_plot: bool = True) -> Tuple[List[ClusterRARResult], np.ndarray, np.ndarray]:
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)

    accept = parse_accept(accept_path)
    results: List[ClusterRARResult] = []
    all_gb = []
    all_gt = []
    for name, shells in accept.items():
        out = run_cluster(name, shells, mu_e=mu_e)
        if out is None:
            continue
        results.append(out)
        all_gb.append(out.g_bar)
        all_gt.append(out.g_tot)

    if not results:
        raise RuntimeError("No ACCEPT clusters matched CLASH sample.")

    gb = np.concatenate(all_gb)
    gt = np.concatenate(all_gt)

    if save_plot:
        fig, ax = plt.subplots(figsize=(6,5), dpi=140)
        ax.scatter(np.log10(gb), np.log10(gt), s=8, alpha=0.55, label='ACCEPT×CLASH shells')
        # one-to-one
        xmin = float(np.nanmin(np.log10(gb)))
        xmax = float(np.nanmax(np.log10(gb)))
        x = np.linspace(xmin, xmax, 256)
        ax.plot(x, x, lw=2, color='k', alpha=0.8, label='g_tot = g_bar')
        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}\;[\,\mathrm{cm\,s^{-2}}\,]$')
        ax.set_ylabel(r'$\log_{10}\,g_{\rm tot}\;[\,\mathrm{cm\,s^{-2}}\,]$')
        ax.set_title('Cluster RAR (gas-only; CLASH NFW total)')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig_path = os.path.join(out_images, 'cluster_rar_scatter.png')
        fig.savefig(fig_path)

    # write points CSV
    csv_path = os.path.join(out_results, 'cluster_rar_points.csv')
    with open(csv_path, 'w') as f:
        f.write('cluster,z,r_kpc,log10_gbar_cgs,log10_gtot_cgs\n')
        for r in results:
            for rkpc, gbv, gtv in zip(r.radii_kpc, r.g_bar, r.g_tot):
                if np.isfinite(gbv) and np.isfinite(gtv) and gbv>0 and gtv>0:
                    f.write(f"{r.cluster},{r.z:.5f},{rkpc:.6f},{math.log10(gbv):.9f},{math.log10(gtv):.9f}\n")

    return results, gb, gt


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build cluster RAR from Umetsu+2016 and ACCEPT data (offline)")
    parser.add_argument('--accept', default=os.path.join('external_data', 'accept_database.dat'), help='Path to ACCEPT database file')
    parser.add_argument('--results', default=os.path.join('results', 'cluster_rar'), help='Output directory for results')
    parser.add_argument('--images', default=os.path.join('images', 'cluster_rar'), help='Output directory for images')
    parser.add_argument('--mu-e', type=float, default=1.17, help='Mean molecular weight per free electron (default 1.17)')
    args = parser.parse_args()

    results, gb, gt = run_all(args.accept, args.results, args.images, mu_e=args.mu_e, save_plot=True)

    # Fit a0 for two illustrative models
    a0_mond, rms_mond = fit_a0(gb, gt, g_pred_mond_simple, a0_min=1e-12, a0_max=1e-7)
    a0_eg, rms_eg = fit_a0(gb, gt, g_pred_emergent, a0_min=1e-12, a0_max=1e-7)

    summary = {
        'cosmology': {'H0_km_s_Mpc': COSMO.H0_km_s_Mpc, 'Omega_m': COSMO.Omega_m, 'Omega_L': COSMO.Omega_L},
        'mu_e': args.mu_e,
        'matched_clusters': len(results),
        'n_points': int(np.isfinite(np.log10(gb)).sum()),
        'a0_fits_cgs': {
            'mond_simple': {'a0': a0_mond, 'rms_dex': rms_mond},
            'eg_like': {'a0': a0_eg, 'rms_dex': rms_eg},
        }
    }
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Console report
    print(f"Fitted a0 (MOND-simple): {a0_mond:.3e} cgs; RMS scatter = {rms_mond:.3f} dex")
    print(f"Fitted a0 (EG-like)    : {a0_eg:.3e} cgs; RMS scatter = {rms_eg:.3f} dex")
    for r in results:
        print(f"{r.cluster:20s} z={r.z:.3f}  r200c={r.nfw.r200c_kpc:7.0f} kpc  c200c={r.nfw.c200c:4.1f}  points={len(r.radii_kpc)}")
