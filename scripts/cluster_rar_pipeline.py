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
# ACCEPT parser and baryon + stars mass builder
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
# Optional BCG/stellar mass plug-in
# ----------------------------
@dataclass
class StarSpec:
    Mstar_Msun: float
    Re_kpc: float
    profile: str = 'hernquist'  # 'hernquist' or 'sersic4' (mapped to hernquist)


def parse_stars_csv(path: str) -> Dict[str, StarSpec]:
    """Parse optional stars CSV: cluster,log10Mstar_BCG,Re_kpc,profile"""
    if not os.path.exists(path):
        return {}
    stars: Dict[str, StarSpec] = {}
    import csv
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('cluster') or row.get('name') or ''
            if not name:
                continue
            try:
                log10M = float(row.get('log10Mstar_BCG') or row.get('log10Mstar') or row.get('log10M'))
                Re_kpc = float(row.get('Re_kpc') or row.get('Re'))
                profile = (row.get('profile') or 'hernquist').strip().lower()
                stars[name] = StarSpec(Mstar_Msun=10**log10M, Re_kpc=Re_kpc, profile=profile)
            except Exception:
                continue
    return stars


def star_M_enclosed_hernquist(r_cm: np.ndarray, Mstar_g: float, Re_kpc: float) -> np.ndarray:
    # Map de Vaucouleurs/Sersic n=4 to Hernquist with a = Re/1.8153
    a_cm = (Re_kpc / 1.8153) * kpc
    r = np.maximum(r_cm, 1e-3)
    return Mstar_g * (r**2) / (r + a_cm)**2


def add_stars_to_baryons(r_cm: np.ndarray,
                         Mgas_cum_g: np.ndarray,
                         star: Optional[StarSpec]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Mbar_cum_g, g_bar_cgs, Mstar_cum_g (for diagnostics)
    """
    if star is None:
        g_bar = G * Mgas_cum_g / (r_cm ** 2)
        return Mgas_cum_g, g_bar, np.zeros_like(Mgas_cum_g)
    Mstar_g = star.Mstar_Msun * M_sun
    Mstar_cum = star_M_enclosed_hernquist(r_cm, Mstar_g, star.Re_kpc)
    Mbar = Mgas_cum_g + Mstar_cum
    g_bar = G * Mbar / (r_cm ** 2)
    return Mbar, g_bar, Mstar_cum


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


def xi_rar_plateau(g_bar: np.ndarray, a0_cgs: float, dmax: float = 50.0) -> np.ndarray:
    """RAR plateau-style boost: xi = min(0.5 + sqrt(0.25 + a0/g_b), D_max)."""
    # guard for divide-by-zero
    gb = np.maximum(g_bar, 1e-99)
    xi = 0.5 + np.sqrt(0.25 + (a0_cgs / gb))
    if dmax is not None and np.isfinite(dmax):
        xi = np.minimum(xi, dmax)
    return xi


def g_pred_rar_plateau(g_bar: np.ndarray, a0_cgs: float, dmax: float = 50.0) -> np.ndarray:
    return xi_rar_plateau(g_bar, a0_cgs, dmax=dmax) * g_bar


def fit_a0(g_bar: np.ndarray,
           g_tot: np.ndarray,
           model: Callable[[np.ndarray, float], np.ndarray],
           a0_min: float = 1e-12, a0_max: float = 1e-6, n_grid: int = 6000) -> Tuple[float, float]:
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


def fit_a0_weighted(
    g_bar: np.ndarray,
    g_tot: np.ndarray,
    model: Callable[[np.ndarray, float], np.ndarray],
    weights: Optional[np.ndarray] = None,
    robust: str = 'none',
    huber_delta: float = 0.2,
    a0_min: float = 1e-12,
    a0_max: float = 1e-6,
    n_grid: int = 6000,
    fixed_a0: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Weighted a0 fit with optional robust loss. Returns (a0, rms_dex, robust_rms_like).
    If fixed_a0 is provided, uses it and only evaluates metrics.
    robust in {'none','huber'}; huber_delta in dex.
    """
    mask = (g_bar > 0) & (g_tot > 0) & np.isfinite(g_bar) & np.isfinite(g_tot)
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan')
    gb = g_bar[mask]
    gt = g_tot[mask]
    if weights is None:
        w = np.ones_like(gb)
    else:
        w = np.asarray(weights)[mask]
        if w.shape != gb.shape:
            raise ValueError('weights shape mismatch')
    lg_gt = np.log10(gt)

    def loss_vals(resid: np.ndarray) -> Tuple[float, float]:
        # returns (mse_rms, robust_metric)
        # weighted MSE RMS
        mse = np.sum(w * (resid ** 2)) / np.sum(w)
        rms = float(np.sqrt(mse))
        if robust == 'huber':
            absr = np.abs(resid)
            quad = 0.5 * np.minimum(absr, huber_delta) ** 2
            lin = huber_delta * (absr - np.minimum(absr, huber_delta))
            hub = quad + lin
            val = float(np.sum(w * hub) / np.sum(w))
            return rms, val
        else:
            return rms, rms

    if fixed_a0 is not None and np.isfinite(fixed_a0):
        pred = model(gb, float(fixed_a0))
        resid = lg_gt - np.log10(pred)
        rms, rrob = loss_vals(resid)
        return float(fixed_a0), rms, rrob

    a0s = np.logspace(math.log10(a0_min), math.log10(a0_max), n_grid)
    best_a0 = None
    best_val = 1e99
    best_rms = 1e99
    for a0 in a0s:
        pred = model(gb, a0)
        resid = lg_gt - np.log10(pred)
        rms, rrob = loss_vals(resid)
        crit = rrob if robust == 'huber' else rms
        if crit < best_val:
            best_val = crit
            best_rms = rms
            best_a0 = a0
    return best_a0 if best_a0 is not None else float('nan'), best_rms, best_val


def compute_residual_metrics(
    clusters: List[str],
    r_kpc: np.ndarray,
    r200c_kpc_by_cluster: Dict[str, float],
    g_bar: np.ndarray,
    g_tot: np.ndarray,
    a0: float,
    dmax: float = 50.0,
    weights_mode: str = 'points',
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute coverage, slope vs x, inner/outer medians, and weights vector.
    weights_mode in {'points','equal_cluster'}.
    Returns (metrics_dict, x, resid, weights)
    """
    x = np.array([rk / r200c_kpc_by_cluster[c] for rk, c in zip(r_kpc, clusters)], dtype=float)
    gb = g_bar.copy()
    gt = g_tot.copy()
    # mask by x range if provided
    mask = np.isfinite(gb) & np.isfinite(gt) & (gb > 0) & (gt > 0)
    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)
    x = x[mask]
    gb = gb[mask]
    gt = gt[mask]
    resid = np.log10(gt) - np.log10(g_pred_rar_plateau(gb, a0, dmax=dmax))
    # weights
    if weights_mode == 'equal_cluster':
        # count per cluster under mask
        from collections import Counter
        masked_clusters = [c for c, m in zip(clusters, mask) if m]
        counts = Counter(masked_clusters)
        w = np.array([1.0 / counts[c] for c in masked_clusters], dtype=float)
    else:
        w = np.ones_like(resid)
    # coverage
    abs_res = np.abs(resid)
    frac_0p1 = float(np.sum(w[abs_res <= 0.1]) / np.sum(w)) if w.size else float('nan')
    frac_0p2 = float(np.sum(w[abs_res <= 0.2]) / np.sum(w)) if w.size else float('nan')
    pos_frac = float(np.sum(w[resid > 0]) / np.sum(w)) if w.size else float('nan')
    # inner/outer medians (unweighted medians are conventional)
    inner = resid[x <= 0.2]
    outer = resid[x > 0.2]
    med_in = float(np.median(inner)) if inner.size else float('nan')
    med_out = float(np.median(outer)) if outer.size else float('nan')
    # linear fit resid ~ a + b x (weighted least squares)
    if x.size >= 2:
        X = np.vstack([np.ones_like(x), x]).T
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ resid)
        a_hat = float(beta[0])
        b_hat = float(beta[1])
        x_zero = float(-a_hat / b_hat) if b_hat != 0 else float('nan')
    else:
        a_hat = b_hat = x_zero = float('nan')
    metrics = {
        'counts': int(resid.size),
        'coverage': {
            'fraction_within_0p1_dex': frac_0p1,
            'fraction_within_0p2_dex': frac_0p2,
            'positive_fraction': pos_frac,
        },
        'radial_trend': {'a': a_hat, 'b': b_hat, 'x_zero': x_zero},
        'medians': {'inner_le_0p2_dex': med_in, 'outer_gt_0p2_dex': med_out},
    }
    return metrics, x, resid, w


def per_cluster_metrics(results: List[ClusterRARResult], a0: float, dmax: float = 50.0) -> List[dict]:
    rows = []
    for r in results:
        gb = r.g_bar
        gt = r.g_tot
        resid_rar = np.log10(gt) - np.log10(g_pred_rar_plateau(gb, a0, dmax=dmax))
        resid_gr = np.log10(gt) - np.log10(g_pred_newton(gb))
        rows.append({
            'cluster': r.cluster,
            'n_points': int(np.sum(np.isfinite(resid_rar))),
            'rms_rar_dex': float(np.sqrt(np.mean(resid_rar**2))),
            'rms_gr_dex': float(np.sqrt(np.mean(resid_gr**2))),
            'median_residual_dex': float(np.median(resid_rar)),
            'mean_residual_dex': float(np.mean(resid_rar)),
        })
    return rows


def jackknife_by_cluster(results: List[ClusterRARResult], weights_mode: str, xmin: Optional[float], xmax: Optional[float], dmax: float = 50.0) -> dict:
    a0_list = []
    for i in range(len(results)):
        sub = results[:i] + results[i+1:]
        clusters = []
        r_kpc = []
        gbar = []
        gtot = []
        r200 = {}
        for r in sub:
            clusters.extend([r.cluster]*len(r.radii_kpc))
            r_kpc.extend(r.radii_kpc.tolist())
            gbar.extend(r.g_bar.tolist())
            gtot.extend(r.g_tot.tolist())
            r200[r.cluster] = r.nfw.r200c_kpc
        clusters = clusters
        r_kpc = np.array(r_kpc)
        gbar = np.array(gbar)
        gtot = np.array(gtot)
        # weights for fit
        _, x, _, w = compute_residual_metrics(clusters, r_kpc, r200, gbar, gtot, a0=1e-8, dmax=dmax, weights_mode=weights_mode, xmin=xmin, xmax=xmax)  # a0 placeholder just to get mask/weights
        # Fit a0 weighted
        a0_i, _, _ = fit_a0_weighted(gbar, gtot, lambda x_, a: g_pred_rar_plateau(x_, a, dmax=dmax), weights=w, robust='none')
        a0_list.append(a0_i)
    arr = np.array(a0_list)
    return {'mean_a0_cgs': float(np.nanmean(arr)), 'std_a0_cgs': float(np.nanstd(arr)), 'n': int(len(arr))}


def bootstrap_rms(results: List[ClusterRARResult], weights_mode: str, xmin: Optional[float], xmax: Optional[float], dmax: float, n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    clusters = []
    r_kpc = []
    gbar = []
    gtot = []
    r200 = {}
    for r in results:
        clusters.extend([r.cluster]*len(r.radii_kpc))
        r_kpc.extend(r.radii_kpc.tolist())
        gbar.extend(r.g_bar.tolist())
        gtot.extend(r.g_tot.tolist())
        r200[r.cluster] = r.nfw.r200c_kpc
    clusters = np.array(clusters, dtype=object)
    r_kpc = np.array(r_kpc)
    gbar = np.array(gbar)
    gtot = np.array(gtot)
    # base weights mask
    metrics_base, x_base, resid_base, w_base = compute_residual_metrics(clusters.tolist(), r_kpc, r200, gbar, gtot, a0=1e-8, dmax=dmax, weights_mode=weights_mode, xmin=xmin, xmax=xmax)
    idx = np.where(np.isfinite(resid_base))[0]
    rms_list = []
    for _ in range(n):
        pick = rng.choice(idx, size=idx.size, replace=True)
        gb = gbar[metrics_base['counts']*0:metrics_base['counts']*0]  # dummy to satisfy linter
        # Build bootstrap sample
        gb = gbar[mask := (np.arange(len(gbar)) == -1)]  # empty, reuse arrays directly
        # Simpler: work with filtered arrays directly
        filt_gb = gbar[mask := (np.arange(len(gbar)) == -1)]  # noop
        lg_gt = np.log10(gbar)  # not used; leave placeholder to avoid heavy copies
        # Use residual_base index to build gb/gt/x/w for picked sample
        gbp = 10**(np.log10(gbar)[idx][pick])
        gtp = 10**(np.log10(gtot)[idx][pick])
        wp = w_base[idx][pick]
        a0_b, rms_b, _ = fit_a0_weighted(gbp, gtp, lambda x_, a: g_pred_rar_plateau(x_, a, dmax=dmax), weights=wp, robust='none')
        rms_list.append(rms_b)
    arr = np.array(rms_list)
    return {'mean_rms_dex': float(np.nanmean(arr)), 'std_rms_dex': float(np.nanstd(arr)), 'n': int(n)}


def make_additional_plots(out_images: str, x: np.ndarray, resid: np.ndarray, alt_dir: Optional[str] = None) -> None:
    os.makedirs(out_images, exist_ok=True)
    # Residuals vs r/R200
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.scatter(x, resid, s=10, alpha=0.6)
    # best-fit line
    if x.size >= 2:
        b, a = np.polyfit(x, resid, 1)
        xs = np.linspace(0, float(np.nanmax(x)), 200)
        ax.plot(xs, a + b*xs, 'r-', lw=1.5, label=f'fit: {a:.3f} + {b:.2f} x')
        ax.legend(frameon=False)
    ax.set_xlabel('r / R200c')
    ax.set_ylabel('Δ log10 g (NFW − RAR)')
    fig.tight_layout()
    p1 = os.path.join(out_images, 'cluster_rar_residuals_vs_r200.png')
    fig.savefig(p1)
    plt.close(fig)
    # Histogram
    fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    ax.hist(resid, bins=30, color='tab:blue', alpha=0.85)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Δ log10 g (NFW − RAR)')
    ax.set_ylabel('count')
    fig.tight_layout()
    p2 = os.path.join(out_images, 'cluster_rar_residual_hist.png')
    fig.savefig(p2)
    plt.close(fig)
    # Alt copies
    if alt_dir:
        try:
            os.makedirs(alt_dir, exist_ok=True)
            import shutil
            shutil.copy(p1, os.path.join(alt_dir, os.path.basename(p1)))
            shutil.copy(p2, os.path.join(alt_dir, os.path.basename(p2)))
        except Exception:
            pass


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


def _validate_and_filter_shells(shells: List[AcceptShell]) -> Tuple[List[AcceptShell], Dict[str, float]]:
    """Basic data hygiene for ACCEPT shells. Returns filtered shells and diagnostics."""
    diags = {
        'n_shells': float(len(shells)),
        'n_used': 0.0,
        'monotonic_ok': 1.0,
        'overlap_removed': 0.0,
        'bad_ne_removed': 0.0,
        'rin_ge_rout_removed': 0.0,
        'min_ne': float('nan'),
        'max_ne': float('nan'),
    }
    out: List[AcceptShell] = []
    last_rout = -1.0
    ne_vals = []
    for sh in shells:
        if not (np.isfinite(sh.Rin_Mpc) and np.isfinite(sh.Rout_Mpc) and np.isfinite(sh.ne_cm3)):
            diags['bad_ne_removed'] += 1.0
            continue
        if sh.Rin_Mpc >= sh.Rout_Mpc:
            diags['rin_ge_rout_removed'] += 1.0
            continue
        # enforce monotonic Rout
        if last_rout > 0 and sh.Rout_Mpc <= last_rout:
            diags['overlap_removed'] += 1.0
            continue
        # very broad sanity band for ne
        if sh.ne_cm3 <= 0 or sh.ne_cm3 > 1e-1:
            diags['bad_ne_removed'] += 1.0
            continue
        out.append(sh)
        last_rout = sh.Rout_Mpc
        ne_vals.append(sh.ne_cm3)
    if len(out) == 0:
        diags['monotonic_ok'] = 0.0
    diags['n_used'] = float(len(out))
    if ne_vals:
        diags['min_ne'] = float(np.min(ne_vals))
        diags['max_ne'] = float(np.max(ne_vals))
    return out, diags


def run_cluster(cluster_name: str,
                accept_shells: List[AcceptShell],
                mu_e: float = 1.17,
                star: Optional[StarSpec] = None) -> Optional[ClusterRARResult]:
    clash_name = _canonicalize_clash_key(cluster_name)
    if clash_name is None:
        return None
    pars = CLASH_PARAMS[clash_name]
    nfw = NFW(M200c_Msun_h70=pars["M200c"], c200c=pars["c200c"], z=pars["z"])

    filtered, _ = _validate_and_filter_shells(accept_shells)
    if not filtered:
        return None
    r_cm, Mgas_cum_g, gbar_gas = build_gas_profiles(filtered, mu_e=mu_e)
    # include optional stars
    _, gbar, _ = add_stars_to_baryons(r_cm, Mgas_cum_g, star)
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
            stars_csv: Optional[str] = None,
            save_plot: bool = True,
            warn_fgas_thresh: float = 0.2,
            warn_used_frac_thresh: float = 0.6) -> Tuple[List[ClusterRARResult], np.ndarray, np.ndarray]:
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)

    accept = parse_accept(accept_path)
    stars_map = parse_stars_csv(stars_csv) if stars_csv else {}
    results: List[ClusterRARResult] = []
    all_gb = []
    all_gt = []
    # diagnostics per-cluster
    diag_rows = []

    for name, shells in accept.items():
        clash_name = _canonicalize_clash_key(name)
        filtered, di = _validate_and_filter_shells(shells)
        if clash_name is None or not filtered:
            continue
        pars = CLASH_PARAMS[clash_name]
        nfw = NFW(M200c_Msun_h70=pars["M200c"], c200c=pars["c200c"], z=pars["z"])
        r_cm, Mgas_cum_g, gbar_gas = build_gas_profiles(filtered, mu_e=mu_e)
        # include optional stars for diagnostics and final g_bar
        star_spec = None
        # try to match stars by provided name or CLASH key
        for key_try in (name, clash_name):
            if key_try in stars_map:
                star_spec = stars_map[key_try]
                break
        Mbar_cum, gbar, Mstar_cum = add_stars_to_baryons(r_cm, Mgas_cum_g, star_spec)
        gtot = nfw.g_tot(r_cm)
        # quick f_bary sanity at 0.5 R200c and R200c
        r200 = nfw.r200c_cm
        def _fratio_at(frac: float, Mnum: np.ndarray) -> float:
            R = frac * r200
            if len(r_cm) == 0:
                return float('nan')
            idx = int(np.clip(np.searchsorted(r_cm, R), 0, len(r_cm)-1))
            Mnum_val = float(Mnum[idx])
            Mtot = float(nfw.M_enclosed(np.array([r_cm[idx]]))[0])
            return Mnum_val / Mtot if Mtot > 0 else float('nan')
        fgas05 = _fratio_at(0.5, Mgas_cum_g)
        fgas1 = _fratio_at(1.0, Mgas_cum_g)
        fbar05 = _fratio_at(0.5, Mbar_cum)
        fbar1 = _fratio_at(1.0, Mbar_cum)
        used_frac = (di['n_used'] / di['n_shells']) if di['n_shells'] > 0 else 0.0
        warn_flags = []
        if fgas1 and fgas1 > warn_fgas_thresh:
            warn_flags.append('FGAS_R200_HIGH')
        if used_frac < warn_used_frac_thresh:
            warn_flags.append('FEW_SHELLS_USED')
        if di['monotonic_ok'] == 0.0:
            warn_flags.append('NON_MONOTONIC_RADII')
        diag_rows.append({
            'cluster': clash_name,
            'z': pars['z'],
            'n_shells': di['n_shells'],
            'n_used': di['n_used'],
            'monotonic_ok': di['monotonic_ok'],
            'overlap_removed': di['overlap_removed'],
            'bad_ne_removed': di['bad_ne_removed'],
            'rin_ge_rout_removed': di['rin_ge_rout_removed'],
            'min_ne': di['min_ne'],
            'max_ne': di['max_ne'],
'r200c_kpc': nfw.r200c_kpc,
            'fgas_0p5R200': fgas05,
            'fgas_R200': fgas1,
            'fbar_0p5R200': fbar05,
            'fbar_R200': fbar1,
            'stars_used': 1 if star_spec is not None else 0,
            'used_frac': used_frac,
            'warn_flags': ';'.join(warn_flags) if warn_flags else ''
        })
        res = ClusterRARResult(clash_name, pars['z'], r_cm/kpc, gbar, gtot, nfw)
        results.append(res)
        all_gb.append(gbar)
        all_gt.append(gtot)

    if not results:
        raise RuntimeError("No ACCEPT clusters matched CLASH sample.")

    gb = np.concatenate(all_gb)
    gt = np.concatenate(all_gt)

    # GR and RAR-plateau predictions for overlay/metrics (global a0)
    g_gr = g_pred_newton(gb)
    a0_rar, rms_rar = fit_a0(gb, gt, lambda x, a0: g_pred_rar_plateau(x, a0, dmax=50.0))

    # Plot
    if save_plot:
        fig, ax = plt.subplots(figsize=(6,5), dpi=140)
        ax.scatter(np.log10(gb), np.log10(gt), s=8, alpha=0.55, label='NFW total (Umetsu+2016)')
        # GR line y=x
        xmin = float(np.nanmin(np.log10(gb)))
        xmax = float(np.nanmax(np.log10(gb)))
        x = np.linspace(xmin, xmax, 256)
        ax.plot(x, x, lw=2, color='k', alpha=0.8, label='GR (baryons only)')
        # RAR plateau curve using global a0
        gx = 10**x
        y_rar = np.log10(g_pred_rar_plateau(gx, a0_rar, dmax=50.0))
        ax.plot(x, y_rar, lw=2, color='tab:red', label=f'RAR plateau (a0={a0_rar:.2e} cgs, Dmax=50)')
        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}\;[\,\mathrm{cm\,s^{-2}}\,]$')
        ax.set_ylabel(r'$\log_{10}\,g_{\rm tot}\;[\,\mathrm{cm\,s^{-2}}\,]$')
        ax.set_title('Cluster RAR: NFW data vs GR and RAR plateau')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig_path = os.path.join(out_images, 'cluster_rar_scatter.png')
        fig.savefig(fig_path)
        # also write a copy to images/next_steps/cluster_rar for docs embedding
        try:
            alt_dir = os.path.join('images', 'next_steps', 'cluster_rar')
            os.makedirs(alt_dir, exist_ok=True)
            fig.savefig(os.path.join(alt_dir, 'cluster_rar_scatter.png'))
        except Exception:
            pass

    # write points CSV with additional columns
    csv_path = os.path.join(out_results, 'cluster_rar_points.csv')
    with open(csv_path, 'w') as f:
        f.write('cluster,z,r_kpc,log10_gbar_cgs,log10_gNFWtot_cgs,log10_gGR_cgs,log10_gRAR_cgs\n')
        for r in results:
            for rkpc, gbv, gtv in zip(r.radii_kpc, r.g_bar, r.g_tot):
                if np.isfinite(gbv) and np.isfinite(gtv) and gbv>0 and gtv>0:
                    gr = g_pred_newton(np.array([gbv]))[0]
                    grar = g_pred_rar_plateau(np.array([gbv]), a0_rar, dmax=50.0)[0]
                    f.write(
                        f"{r.cluster},{r.z:.5f},{rkpc:.6f},{math.log10(gbv):.9f},{math.log10(gtv):.9f},{math.log10(gr):.9f},{math.log10(grar):.9f}\n"
                    )

    # diagnostics CSV
    diag_csv = os.path.join(out_results, 'diagnostics.csv')
    with open(diag_csv, 'w') as f:
        cols = ['cluster','z','n_shells','n_used','monotonic_ok','overlap_removed','bad_ne_removed','rin_ge_rout_removed','min_ne','max_ne','r200c_kpc','fgas_0p5R200','fgas_R200','fbar_0p5R200','fbar_R200','stars_used','used_frac','warn_flags']
        f.write(','.join(cols) + '\n')
        for row in diag_rows:
            f.write(','.join(str(row.get(c, '')) for c in cols) + '\n')

    # metrics JSON
    mask = (gb>0) & (gt>0)
    metrics = {
        'counts': int(mask.sum()),
        'rms_logdex': {
            'GR': float(np.sqrt(np.mean((np.log10(gt[mask]) - np.log10(g_gr[mask]))**2))) if mask.any() else float('nan'),
            'RAR_plateau': float(rms_rar),
        },
        'rar_plateau': {'a0_cgs': float(a0_rar), 'Dmax': 50.0},
    }
    with open(os.path.join(out_results, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

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
    parser.add_argument('--stars-csv', default=os.path.join('external_data', 'clash_stars.csv'), help='Optional CSV with BCG stellar masses (cluster,log10Mstar_BCG,Re_kpc,profile)')
    parser.add_argument('--warn-fgas', type=float, default=0.2, help='Warn if f_gas(R200) exceeds this threshold (default 0.2)')
parser.add_argument('--warn-used-frac', type=float, default=0.6, help='Warn if n_used/n_shells below this fraction (default 0.6)')
    # Analysis enhancements
    parser.add_argument('--equal-cluster-weight', action='store_true', help='Weight each cluster equally (each point in a cluster gets weight 1/N_cluster)')
    parser.add_argument('--xmin', type=float, default=None, help='Mask out points with r/R200c < xmin')
    parser.add_argument('--xmax', type=float, default=None, help='Mask out points with r/R200c > xmax')
    parser.add_argument('--robust-loss', choices=['none','huber'], default='none', help='Loss for a0 fit')
    parser.add_argument('--huber-delta', type=float, default=0.2, help='Huber delta (dex) for robust loss')
    parser.add_argument('--fixed-a0', type=float, default=None, help='Fix a0 (cgs) instead of fitting')
    parser.add_argument('--jackknife-by-cluster', action='store_true', help='Leave-one-cluster-out jackknife for a0')
    parser.add_argument('--bootstrap-points', type=int, default=0, help='Bootstrap replicates for RMS (0 disables)')
    parser.add_argument('--null-tests', action='store_true', help='Run null tests (radial scramble, cross-match scramble)')
    args = parser.parse_args()

    results, gb, gt = run_all(args.accept, args.results, args.images, mu_e=args.mu_e, stars_csv=args.stars_csv, save_plot=True, warn_fgas_thresh=args.warn_fgas, warn_used_frac_thresh=args.warn_used_frac)

    # Fit a0 for two illustrative models (global)
    a0_mond, rms_mond = fit_a0(gb, gt, g_pred_mond_simple)
    a0_eg, rms_eg = fit_a0(gb, gt, g_pred_emergent)
    # Build arrays per point with cluster mapping for enhanced metrics
    clusters = []
    r_kpc = []
    gbar_list = []
    gtot_list = []
    r200_map = {}
    for r in results:
        clusters.extend([r.cluster]*len(r.radii_kpc))
        r_kpc.extend(r.radii_kpc.tolist())
        gbar_list.extend(r.g_bar.tolist())
        gtot_list.extend(r.g_tot.tolist())
        r200_map[r.cluster] = r.nfw.r200c_kpc
    clusters = clusters
    r_kpc = np.array(r_kpc)
    gbar_all = np.array(gbar_list)
    gtot_all = np.array(gtot_list)
    weights_mode = 'equal_cluster' if args.equal_cluster_weight else 'points'

    # Compute weights and fit a0 (RAR plateau) with options
    # First compute weights mask via metrics helper (a0 placeholder is fine for mask/weights)
    base_metrics, x_base, resid_base, w_base = compute_residual_metrics(clusters, r_kpc, r200_map, gbar_all, gtot_all, a0=1e-8, dmax=50.0, weights_mode=weights_mode, xmin=args.xmin, xmax=args.xmax)
    a0_rar, rms_rar, _ = fit_a0_weighted(gbar_all, gtot_all, lambda x, a: g_pred_rar_plateau(x, a, dmax=50.0), weights=w_base, robust=args.robust_loss, huber_delta=args.huber_delta, fixed_a0=args.fixed_a0)

    summary = {
        'cosmology': {'H0_km_s_Mpc': COSMO.H0_km_s_Mpc, 'Omega_m': COSMO.Omega_m, 'Omega_L': COSMO.Omega_L},
        'mu_e': args.mu_e,
        'matched_clusters': len(results),
        'n_points': int(np.isfinite(np.log10(gb)).sum()),
'a0_fits_cgs': {
            'mond_simple': {'a0': a0_mond, 'rms_dex': rms_mond},
            'eg_like': {'a0': a0_eg, 'rms_dex': rms_eg},
            'rar_plateau': {'a0': a0_rar, 'Dmax': 50.0, 'rms_dex': rms_rar},
        }
    }
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Enhanced metrics and plots (defensible analysis pieces)
    # Residual metrics with final a0
    resid_metrics, x, resid, w = compute_residual_metrics(clusters, r_kpc, r200_map, gbar_all, gtot_all, a0=a0_rar, dmax=50.0, weights_mode=weights_mode, xmin=args.xmin, xmax=args.xmax)
    # Additional plots
    alt_dir = os.path.join('images', 'next_steps', 'cluster_rar')
    make_additional_plots(args.images, x, resid, alt_dir=alt_dir)

    # Per-cluster CSV
    per_rows = per_cluster_metrics(results, a0=a0_rar, dmax=50.0)
    with open(os.path.join(args.results, 'cluster_section_per_cluster.csv'), 'w') as f:
        f.write('cluster,n_points,rms_rar_dex,rms_gr_dex,median_residual_dex,mean_residual_dex\r\n')
        for row in per_rows:
            f.write(f"{row['cluster']},{row['n_points']},{row['rms_rar_dex']},{row['rms_gr_dex']},{row['median_residual_dex']},{row['mean_residual_dex']}\r\n")

    # Aggregated metrics JSON (with jackknife/bootstrap if requested)
    agg = {
        'counts': resid_metrics['counts'],
        'rar_plateau': {
            'a0_cgs': float(a0_rar),
            'Dmax': 50.0,
            'rms_dex': float(rms_rar),
        },
        'gr': {
            'rms_dex': float(np.sqrt(np.mean((np.log10(gt[gt>0]) - np.log10(gb[gb>0]))**2))) if (gt>0).any() else float('nan'),
            'median_residual_dex': float(np.median(np.log10(gt[gt>0]) - np.log10(gb[gb>0]))) if (gt>0).any() else float('nan'),
            'mean_residual_dex': float(np.mean(np.log10(gt[gt>0]) - np.log10(gb[gb>0]))) if (gt>0).any() else float('nan'),
        },
        'coverage': resid_metrics['coverage'],
        'radial_trend': resid_metrics['radial_trend'],
        'medians': resid_metrics['medians'],
    }
    # Jackknife
    if args.jackknife_by_cluster:
        agg['rar_plateau']['jackknife'] = jackknife_by_cluster(results, weights_mode=weights_mode, xmin=args.xmin, xmax=args.xmax, dmax=50.0)
    # Bootstrap
    if args.bootstrap_points and args.bootstrap_points > 0:
        agg['rar_plateau']['bootstrap_rms'] = bootstrap_rms(results, weights_mode=weights_mode, xmin=args.xmin, xmax=args.xmax, dmax=50.0, n=args.bootstrap_points)
    # Null tests
    if args.null_tests:
        import numpy as _np
        # Radial scramble (shuffle x within cluster)
        scrambled_x = x.copy()
        # recompute x per-cluster strictly
        x_all = np.array([rk / r200_map[c] for rk, c in zip(r_kpc, clusters)], dtype=float)
        scrambled = []
        for cl in set(clusters):
            idx = [i for i,(c,mask) in enumerate(zip(clusters, np.isfinite(x_all))) if c==cl]
            vals = x_all[idx]
            _np.random.shuffle(vals)
            for k,v in zip(idx, vals):
                scrambled.append((k,v))
        x_scr = x_all.copy()
        for k,v in scrambled:
            x_scr[k] = v
        # filter mask
        mask_x = np.ones_like(x_scr, dtype=bool)
        if args.xmin is not None:
            mask_x &= (x_scr >= args.xmin)
        if args.xmax is not None:
            mask_x &= (x_scr <= args.xmax)
        # Use same a0, compute slope on scrambled x
        x_scr2 = x_scr[mask_x & np.isfinite(gbar_all) & np.isfinite(gtot_all) & (gbar_all>0) & (gtot_all>0)]
        resid_scr = (np.log10(gtot_all) - np.log10(g_pred_rar_plateau(gbar_all, a0_rar, dmax=50.0)))[mask_x & np.isfinite(gbar_all) & np.isfinite(gtot_all) & (gbar_all>0) & (gtot_all>0)]
        if x_scr2.size >= 2:
            b_s, a_s = np.polyfit(x_scr2, resid_scr, 1)
        else:
            a_s = b_s = float('nan')
        # Cross-match scramble (permute g_tot across points)
        perm = _np.random.permutation(gtot_all.size)
        gt_perm = gtot_all[perm]
        a0_perm, rms_perm, _ = fit_a0_weighted(gbar_all, gt_perm, lambda xx, aa: g_pred_rar_plateau(xx, aa, dmax=50.0), weights=w_base, robust='none')
        agg['null_tests'] = {
            'radial_scramble': {'a': float(a_s), 'b': float(b_s)},
            'cross_match_scramble': {'a0_cgs': float(a0_perm), 'rms_dex': float(rms_perm)},
        }

    with open(os.path.join(args.results, 'cluster_section_metrics.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    # Console report
    print(f"Fitted a0 (MOND-simple): {a0_mond:.3e} cgs; RMS scatter = {rms_mond:.3f} dex")
    print(f"Fitted a0 (EG-like)    : {a0_eg:.3e} cgs; RMS scatter = {rms_eg:.3f} dex")
    print(f"Fitted a0 (RAR plateau): {a0_rar:.3e} cgs; RMS scatter = {rms_rar:.3f} dex")
    for r in results:
        print(f"{r.cluster:20s} z={r.z:.3f}  r200c={r.nfw.r200c_kpc:7.0f} kpc  c200c={r.nfw.c200c:4.1f}  points={len(r.radii_kpc)}")
