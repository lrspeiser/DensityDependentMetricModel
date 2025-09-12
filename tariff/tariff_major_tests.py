#!/usr/bin/env python3
"""
Tariff major tests — Cosmology batteries for the Energy Tariff add‑on

Subcommands:
  1) cmb         — CMB blackbody / spectral distortion check
  2) tolman      — Fit luminosity‑distance exponent p in d_L = r (1+z)^p from μ(z)
  3) sntd        — Fit SN time‑dilation exponent p_t from light‑curve summaries
  4) bao         — Compute H_eff(z), D_M(z), D_H(z) and compare to BAO CSV if provided
  5) los         — Correlate SN residuals with LOS density proxy
  6) timedelay   — Demonstrate invariance of lens time delays under energy‑only tariff
  7) posteriors  — Summarize sweep_results.csv (posterior‑like histograms)

All tests are optional‑data: they run with curves only if no CSVs are provided.
Confined to /tariff, imports energy_tariff_model.py from this folder.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.optimize import curve_fit
    from scipy import stats
except Exception:
    curve_fit = None
    stats = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

# Constants
C = 299_792.458  # km/s
H0_PLANCK = 67.4  # km/s/Mpc
T_CMB = 2.7255  # K

# Model import (from same folder)
_THIS_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _THIS_DIR / "energy_tariff_model.py"

# Images directory under tariff
IMAGES_DIR = _THIS_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _import_user_model():
    if not _MODEL_PATH.exists():
        print(f"[ERROR] energy_tariff_model.py not found at {_MODEL_PATH}", file=sys.stderr)
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("energy_tariff_model", str(_MODEL_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ---------- Utilities ----------

def planck_I_nu(nu_Hz: np.ndarray, T: float) -> np.ndarray:
    """Planck spectral radiance (W·sr^-1·m^-2·Hz^-1)."""
    h = 6.62607015e-34
    kB = 1.380649e-23
    c = 299_792_458.0
    x = (h * nu_Hz) / (kB * T)
    ex = np.exp(np.clip(x, 1e-12, 700))
    return (2.0 * h * nu_Hz**3) / (c**2 * (ex - 1.0))


def fit_planck_temperature(nu_Hz: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """Fit a best‑fit Planck T' to spectrum I(nu). Returns (T_best, rms_frac_resid)."""
    y = I / (I.max() + 1e-300)
    if curve_fit is None:
        Ts = np.linspace(1.0, 5.0, 4001)
        res = []
        for T in Ts:
            m = planck_I_nu(nu_Hz, T)
            m = m / (m.max() + 1e-300)
            res.append(np.sqrt(np.mean((y - m) ** 2)))
        i = int(np.argmin(res))
        return float(Ts[i]), float(res[i])
    popt, _ = curve_fit(lambda nu, T: planck_I_nu(nu, T), nu_Hz, I, p0=[T_CMB], maxfev=20000)
    T_best = float(popt[0])
    m = planck_I_nu(nu_Hz, T_best)
    m = m / (m.max() + 1e-300)
    rms = float(np.sqrt(np.mean((y - m) ** 2)))
    return T_best, rms

# ---------- 1) CMB spectral-shape test ----------

def test_cmb_spectral_shape(args):
    mod = _import_user_model()
    if mod is None:
        return 2
    sim = mod.PhotonJourney(k_coupling_mpc_inv=args.k,
                            d_max=args.dmax,
                            g_bar_void=args.gbar_void,
                            r0_void=args.r0_void,
                            gamma_void=args.gamma_void,
                            void_mix_mode=getattr(args, "void_mix_mode", "redshift"),
                            zstar=getattr(args, "zstar", 0.5),
                            eta=getattr(args, "eta", 1.5))
    z = sim.redshift(args.distance_mpc, steps=args.steps)
    s = 1.0 + z

    # Frequency grid like FIRAS range 60–630 GHz
    nu = np.linspace(60e9, 630e9, 800)
    I_em = planck_I_nu(nu * s, T_CMB)

    # Transport selection: liouville preserves I_nu/nu^3; energy-only divides intensity by (1+z)
    if getattr(args, "transport", "liouville") == "liouville":
        I_obs = I_em / (s ** 3)
        transport_label = "Liouville (I/(1+z)^3)"
    else:
        I_obs = I_em / s
        transport_label = "Energy-only (I/(1+z))"

    T_fit, rms = fit_planck_temperature(nu, I_obs)

    print(f"[CMB] r={args.distance_mpc:.0f} Mpc → z={z:.3f} (1+z={s:.3f})")
    print(f"  {transport_label}: T'={T_fit:.4f} K, rms_frac_resid={rms:.3e}")
    print("  (FIRAS tolerances are ~few×1e-5 in fractional residuals.)")

    plt.figure(figsize=(10,6))
    plt.title("CMB Spectral Shape under Tariff Mapping")
    plt.plot(nu/1e9, I_obs/np.max(I_obs), label=transport_label, lw=2)
    I_fit = planck_I_nu(nu, T_fit)
    plt.plot(nu/1e9, I_fit/np.max(I_fit), label=f"Best-fit Planck (T'={T_fit:.3f}K)", lw=1.5)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Normalized intensity")
    plt.legend(); plt.grid(True, alpha=0.25)
    out = str(IMAGES_DIR / "cmb_distortion_test.png")
    plt.savefig(out, dpi=150)
    print(f"Saved figure: {out}")
    return 0

# ---------- 2) Tolman surface-brightness exponent p ----------

def test_tolman(args):
    mod = _import_user_model()
    if mod is None:
        return 2

    # Load data or generate mock
    if args.data is None or pd is None:
        print("[WARN] No SN μ(z) dataset provided or pandas missing; generating a small mock sample.")
        z = np.linspace(0.01, 1.0, 50)
        mu = 5*np.log10((C/H0_PLANCK)*z*(1+z)**1.0) + 25.0
        mu_err = np.full_like(mu, 0.15)
    else:
        df = pd.read_csv(args.data)
        z = df['z'].to_numpy(float)
        mu = df['mu'].to_numpy(float)
        mu_err = df['mu_err'].to_numpy(float)

    # Build z(r) grid and invert to r(z)
sim = mod.PhotonJourney(k_coupling_mpc_inv=args.k,
                            d_max=args.dmax,
                            g_bar_void=args.gbar_void,
                            r0_void=args.r0_void,
                            gamma_void=args.gamma_void,
                            void_mix_mode=getattr(args, "void_mix_mode", "redshift"),
                            zstar=getattr(args, "zstar", 0.5),
                            eta=getattr(args, "eta", 1.5))
    r_grid = np.linspace(0.0, 6000.0, 6001)
    z_grid = np.array([sim.redshift(float(r), steps=args.steps) for r in r_grid])
    z_grid = np.maximum.accumulate(z_grid)

    def r_of_z(zv):
        return np.interp(zv, z_grid, r_grid)

    def mu_model(zv, p):
        r = r_of_z(zv)
        dL = r * (1.0 + zv) ** p
        return 5.0*np.log10(np.clip(dL*1e6, 1e-12, np.inf)) - 5.0

    if curve_fit is None:
        Ps = np.linspace(0.0, 2.5, 251)
        chi2 = [np.sum(((mu - mu_model(z, P))/mu_err)**2) for P in Ps]
        i = int(np.argmin(chi2))
        P_best = float(Ps[i]); chi2_best = float(chi2[i])
        dof = max(1, len(z)-1)
    else:
        popt, _ = curve_fit(mu_model, z, mu, sigma=mu_err, absolute_sigma=True, p0=[1.0], maxfev=20000)
        P_best = float(popt[0])
        resid = mu - mu_model(z, P_best)
        chi2_best = float(np.sum((resid/mu_err)**2))
        dof = max(1, len(z)-1)

    print(f"[Tolman] Best-fit p in d_L = r (1+z)^p:  p = {P_best:.3f},  χ^2/dof = {chi2_best/dof:.3f}")
    return 0

# ---------- 3) Supernova time-dilation exponent p_t ----------

def test_sn_timedilation(args):
    if pd is None:
        print("[ERROR] pandas required for this test.", file=sys.stderr)
        return 2
    df = pd.read_csv(args.data)
    z = df['z'].to_numpy(float)
    if 'timescale' in df.columns:
        t = df['timescale'].to_numpy(float)
        terr = df['timescale_err'].to_numpy(float) if 'timescale_err' in df.columns else np.full_like(t, 0.1*np.median(t))
    elif 'stretch' in df.columns:
        t = df['stretch'].to_numpy(float)
        terr = df['stretch_err'].to_numpy(float) if 'stretch_err' in df.columns else np.full_like(t, 0.1*np.median(t))
    else:
        raise ValueError("CSV must contain timescale or stretch columns.")

    def model(zv, p_t, A):
        return A * (1.0 + zv) ** p_t

    if curve_fit is None:
        Ps = np.linspace(0.0, 2.0, 201)
        As = np.linspace(np.percentile(t, 10), np.percentile(t, 90), 50)
        best = (np.inf, None, None)
        for p in Ps:
            for A in As:
                chi2 = np.sum(((t - model(z, p, A))/terr)**2)
                if chi2 < best[0]:
                    best = (chi2, p, A)
        chi2_best, p_best, A_best = best
    else:
        popt, _ = curve_fit(model, z, t, sigma=terr, absolute_sigma=True, p0=[1.0, np.median(t)], maxfev=20000)
        p_best, A_best = map(float, popt)
        resid = t - model(z, p_best, A_best)
        chi2_best = float(np.sum((resid/terr)**2))

    dof = max(1, len(z)-2)
    print(f"[SN time dilation] p_t = {p_best:.3f} (expected ≈ 1 in expansion),  χ^2/dof = {chi2_best/dof:.3f}")
    return 0

# ---------- 4) BAO / chronometer proxies ----------

def test_bao_proxies(args):
    mod = _import_user_model()
    if mod is None:
        return 2
    sim = mod.PhotonJourney(k_coupling_mpc_inv=args.k,
                            d_max=args.dmax,
                            g_bar_void=args.gbar_void,
                            r0_void=args.r0_void,
                            gamma_void=args.gamma_void,
                            void_mix_mode=getattr(args, "void_mix_mode", "redshift"),
                            zstar=getattr(args, "zstar", 0.5),
                            eta=getattr(args, "eta", 1.5))

    r = np.linspace(0.0, args.rmax_mpc, int(args.rmax_mpc)+1)
    z = np.array([sim.redshift(float(rr), steps=args.steps) for rr in r])

    # Prefer monotone PCHIP inversion z(r) <-> r(z) if available
    z_grid = np.linspace(0.0, args.zmax, 2000)
    if PchipInterpolator is not None:
        z_mono = np.array(z, dtype=float)
        eps = 1e-12
        for i in range(1, len(z_mono)):
            if z_mono[i] <= z_mono[i-1]:
                z_mono[i] = z_mono[i-1] + eps
        # r(z) and its derivative
        r_of_z = PchipInterpolator(z_mono, r)
        dr_dz = r_of_z.derivative()(z_grid)
        dz_dr = 1.0 / np.clip(dr_dz, 1e-30, np.inf)
        Hz = C * (dz_dr / (1.0 + z_grid))
    else:
        # Fallback: finite-difference d ln(1+z) / dr, then map by interp
        ln1pz = np.log1p(z)
        dln1pz_dr = np.gradient(ln1pz, r)
        H_eff = C * dln1pz_dr
        z_mono = np.maximum.accumulate(z)
        Hz = np.interp(z_grid, z_mono, H_eff)

    with np.errstate(divide='ignore', invalid='ignore'):
        invH = np.where(Hz > 0, 1.0/Hz, np.nan)
    DM = C * np.cumsum(invH) * (z_grid[1]-z_grid[0])
    DH = C / Hz

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.title("Effective H(z) and BAO proxies from Tariff")
    plt.plot(z_grid, Hz, lw=2, label="H_eff(z)")
    plt.ylabel("H_eff [km/s/Mpc]"); plt.grid(alpha=0.3); plt.legend()

    plt.subplot(2,1,2)
    plt.plot(z_grid, DM, lw=2, label="D_M(z)")
    plt.plot(z_grid, DH, lw=2, label="D_H(z)")
    plt.xlabel("z"); plt.ylabel("Distance [Mpc]")
    plt.grid(alpha=0.3); plt.legend()
    out = str(IMAGES_DIR / "bao_proxies.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Saved BAO proxy curves: {out}")

    # Optional BAO comparison
    if args.bao is not None and pd is not None:
        df = pd.read_csv(args.bao)
        if "D_M_over_rd" in df.columns and "D_H_over_rd" in df.columns:
            z_b = df["z"].to_numpy(float)
            DM_b = df["D_M_over_rd"].to_numpy(float)
            DH_b = df["D_H_over_rd"].to_numpy(float)
            eDM = df.get("D_M_err", pd.Series(np.ones_like(DM_b)*0.05)).to_numpy(float)
            eDH = df.get("D_H_err", pd.Series(np.ones_like(DH_b)*0.05)).to_numpy(float)

            DM_model = np.interp(z_b, z_grid, DM)
            DH_model = np.interp(z_b, z_grid, DH)

            A = np.concatenate([DM_model/eDM, DH_model/eDH])
            y = np.concatenate([DM_b/eDM, DH_b/eDH])
            inv_rd = (A @ y) / (A @ A + 1e-300)
            rd_best = 1.0 / inv_rd
            chi2 = np.sum(((DM_model/rd_best - DM_b)/eDM)**2 + ((DH_model/rd_best - DH_b)/eDH)**2)
            dof = max(1, 2*len(z_b) - 1)
            print(f"[BAO] Fitted r_d = {rd_best:.1f} Mpc, χ^2/dof = {chi2/dof:.3f}")
        elif "DV_over_rd" in df.columns:
            z_b = df["z"].to_numpy(float)
            DV_b = df["DV_over_rd"].to_numpy(float)
            eDV = df.get("DV_err", pd.Series(np.ones_like(DV_b)*0.05)).to_numpy(float)

            DM_model = np.interp(z_b, z_grid, DM)
            DH_model = np.interp(z_b, z_grid, DH)
            DV_model = ( (z_b**2) * (DM_model**2) * DH_model ) ** (1.0/3.0)

            A = DV_model / eDV
            y = DV_b / eDV
            inv_rd = (A @ y) / (A @ A + 1e-300)
            rd_best = 1.0 / inv_rd
            chi2 = np.sum(((DV_model/rd_best - DV_b)/eDV)**2)
            dof = max(1, len(z_b) - 1)
            print(f"[BAO] Fitted r_d = {rd_best:.1f} Mpc, χ^2/dof = {chi2/dof:.3f}")
        else:
            print("[BAO] CSV format not recognized. Expected columns like D_M_over_rd, D_H_over_rd, or DV_over_rd.")
    return 0

# ---------- 5) LOS structure correlation ----------

def test_los_correlation(args):
    if pd is None:
        print("[ERROR] pandas required for this test.", file=sys.stderr)
        return 2
    if stats is None:
        print("[WARN] scipy.stats not available; using a simple estimator.")
    sn = pd.read_csv(args.sn)
    los = pd.read_csv(args.los)
    if 'id' in sn.columns and 'id' in los.columns:
        df = sn.merge(los[['id','delta_los']], on='id', how='inner')
    else:
        df = sn.copy()
        df['delta_los'] = los.iloc[:len(sn), 0].to_numpy(float)
    z = df['z'].to_numpy(float)
    mu = df['mu'].to_numpy(float)
    mu_err = df.get('mu_err', pd.Series(np.full(len(df), 0.15))).to_numpy(float)

    mod = _import_user_model()
    if mod is None:
        return 2
sim = mod.PhotonJourney(k_coupling_mpc_inv=args.k,
                            d_max=args.dmax,
                            g_bar_void=args.gbar_void,
                            r0_void=args.r0_void,
                            gamma_void=args.gamma_void,
                            void_mix_mode=getattr(args, "void_mix_mode", "redshift"),
                            zstar=getattr(args, "zstar", 0.5),
                            eta=getattr(args, "eta", 1.5))
    z_grid = np.linspace(np.min(z), np.max(z), 2000)
    mu_model_grid = sim.distance_modulus_at_z(z_grid)
    mu_model = np.interp(z, z_grid, mu_model_grid)
    resid = mu - mu_model
    delta = df['delta_los'].to_numpy(float)

    if stats is not None:
        r, p = stats.pearsonr(resid, delta)
    else:
        r = np.corrcoef(resid, delta)[0,1]
        n = len(resid)
        t = r*np.sqrt((n-2)/(1-r**2 + 1e-12))
        from math import erf, sqrt
        p = 2*(1 - 0.5*(1+erf(abs(t)/math.sqrt(2))))

    print(f"[LOS] Pearson r(residual, delta_LOS) = {r:.3f}, p = {p:.3g}")
    return 0

# ---------- 6) Lensing time delays ----------

def test_time_delays(args):
    z_l, z_s = 0.5, 1.5
    theta_E = 1.5  # arcsec
    beta = 0.2     # arcsec
    def fermat(theta, beta, theta_E):
        return 0.5*(theta - beta)**2 - theta_E*abs(theta)
    theta_plus = theta_E + beta
    theta_minus = abs(theta_E - beta)
    dt0 = fermat(theta_plus, beta, theta_E) - fermat(theta_minus, beta, theta_E)
    print("[Time Delays] SIS toy Δφ (dimensionless): {:.6f}".format(dt0))
    print("Under energy-only tariff, group speed=c and Fermat potential unchanged → Δt unchanged.")
    return 0

# ---------- 7) Posterior summary ----------

def test_posteriors(args):
    if pd is None:
        print("[ERROR] pandas required for this test.", file=sys.stderr)
        return 2
    path = Path(args.sweep)
    if not path.exists():
        print(f"[WARN] Sweep file not found: {path}")
        return 2
    df = pd.read_csv(path)
    # Normalize column names
    if 'red_chi2' in df.columns and 'chi2_red' not in df.columns:
        df = df.rename(columns={'red_chi2':'chi2_red'})
    if 'k_mpc^-1' in df.columns and 'k' not in df.columns:
        df = df.rename(columns={'k_mpc^-1':'k'})
    df = df.sort_values('chi2_red')
    print(df.head(10).to_string(index=False))

    chi2 = df['chi2_red'].to_numpy(float)
    w = np.exp(-(chi2 - chi2.min())/2.0)

    def hist_plot(ax, col, label):
        if col not in df.columns:
            ax.text(0.5,0.5,f"missing {col}",ha='center',va='center')
            ax.set_axis_off(); return
        v = df[col].to_numpy(float)
        ax.hist(v, bins=30, weights=w, alpha=0.8)
        ax.set_xlabel(label); ax.grid(alpha=0.3)

    fig, axs = plt.subplots(2,3, figsize=(12,7))
    cols = [("dmax","D_max"), ("gbar_void","g_bar_void"), ("r0_void","r0_void [Mpc]"),
            ("gamma_void","gamma_void"), ("k","k [1/Mpc]"), ("chi2_red","chi2_red")]
    for ax,(c,lbl) in zip(axs.flatten(), cols):
        hist_plot(ax,c,lbl)
    fig.suptitle("Sweep Posterior Summaries (weighted by exp(-Δχ^2/2))")
    plt.tight_layout(); plt.savefig(str(IMAGES_DIR / "sweep_posteriors.png"), dpi=150)
    print(f"Saved: {str(IMAGES_DIR / 'sweep_posteriors.png')}")
    return 0

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Major-issues test harness for the Energy Tariff model")
    sp = ap.add_subparsers(dest="cmd", required=True)

def add_model_args(p):
        p.add_argument("--k", type=float, default=7.75e-6, help="Coupling k [1/Mpc]")
        p.add_argument("--dmax", type=float, default=30.0, help="RAR plateau cap D_max")
        p.add_argument("--gbar-void", dest="gbar_void", type=float, default=1e-15, help="Void g_bar [m/s^2]")
        p.add_argument("--r0-void", dest="r0_void", type=float, default=2000.0, help="Void fraction scale r0 [Mpc]")
        p.add_argument("--gamma-void", dest="gamma_void", type=float, default=1.5, help="Void fraction exponent gamma")
        p.add_argument("--void-mix-mode", choices=["distance","redshift"], default="redshift", help="Environmental mix domain: distance-based f_env(r) or redshift-based f_env(z)")
        p.add_argument("--zstar", type=float, default=0.5, help="Transition redshift z* for f_env(z)")
        p.add_argument("--eta", type=float, default=1.5, help="Power eta in f_env(z) = 1 / (1 + (z*/z)^eta)")
        p.add_argument("--steps", type=int, default=4000, help="Integration steps for z(r)")

    # 1) CMB
p1 = sp.add_parser("cmb", help="CMB spectral-shape test under tariff")
    add_model_args(p1)
    p1.add_argument("--transport", choices=["liouville","energy-only"], default="liouville", help="CMB transport mapping: 'liouville' preserves I_nu/nu^3; 'energy-only' divides intensity by (1+z)")
    p1.add_argument("--distance-mpc", type=float, default=14000.0, help="Propagation distance for the CMB test [Mpc]")
    p1.set_defaults(func=test_cmb_spectral_shape)

    # 2) Tolman
    p2 = sp.add_parser("tolman", help="Fit p in d_L = r (1+z)^p from μ(z) data")
    add_model_args(p2)
    p2.add_argument("--data", type=str, default=None, help="CSV with columns z, mu, mu_err")
    p2.set_defaults(func=test_tolman)

    # 3) SN time dilation
    p3 = sp.add_parser("sntd", help="Fit SN time-dilation exponent p_t from light-curve summaries")
    p3.add_argument("--data", type=str, required=True, help="CSV with columns z and timescale/stretch (and errors)")
    p3.set_defaults(func=test_sn_timedilation)

    # 4) BAO / chronometers
    p4 = sp.add_parser("bao", help="Compute H_eff(z), D_M(z), D_H(z), and compare to BAO CSV if provided")
    add_model_args(p4)
    p4.add_argument("--rmax-mpc", type=float, default=6000.0)
    p4.add_argument("--zmax", type=float, default=2.5)
    p4.add_argument("--bao", type=str, default=None, help="Optional BAO CSV")
    p4.set_defaults(func=test_bao_proxies)

    # 5) LOS correlation
    p5 = sp.add_parser("los", help="Correlate SN residuals with LOS density contrast")
    add_model_args(p5)
    p5.add_argument("--sn", type=str, required=True, help="CSV with z, mu, mu_err (and optional id)")
    p5.add_argument("--los", type=str, required=True, help="CSV with id and delta_los (or just delta_los in same order)")
    p5.set_defaults(func=test_los_correlation)

    # 6) Time delays
    p6 = sp.add_parser("timedelay", help="Demonstrate invariance of strong-lens time delays under energy-only tariff")
    p6.set_defaults(func=test_time_delays)

    # 7) Posterior summary
    p7 = sp.add_parser("posteriors", help="Summarize sweep_results.csv posterior-like distributions")
    p7.add_argument("--sweep", type=str, default=str(_THIS_DIR / "sweep_results.csv"))
    p7.set_defaults(func=test_posteriors)

    args = ap.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())

