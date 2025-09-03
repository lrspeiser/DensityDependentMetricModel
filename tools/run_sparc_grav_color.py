#!/usr/bin/env python3
"""
run_sparc_grav_color.py - Evaluate modified-gravity gates on SPARC galaxies
and compare against observed rotation curves (no dark matter).

Model:
  v_model(R) = v_baryon(R) * sqrt(xi(R))

Gate options (same 3 knobs: pivot, gamma, lambda_g):
  - rho  (grav_color):
      xi(R) = 1 + lambda_g / (1 + (rho_mid(R)/rho_c)^gamma)
  - gbar (rar_gate):
      xi(R) = 1 + lambda_g / (1 + (gbar(R)/a0)^gamma),  gbar = vbar^2/R (SI)
  - sigma (surface-density gate):
      xi(R) = 1 + lambda_g / (1 + (Sigma_b(R)/sigma_c)^gamma),  Sigma_b in Msun/pc^2

Inputs from SPARC:
  - v_baryon built from SPARC "rotmod" components: V_gas, V_disk, V_bulge
  - rho_mid from SPARC surface densities and assumed scale heights via sparc_io
  - Sigma_b from rotmod SB columns (converted to Msun/pc^2) plus gas Sigma (Msun/pc^2)

Defaults for xi parameters are taken from the Milky Way 144k grav_color fit
observed earlier (rounded):
  rho_c ~= 3.48e8 Msun/kpc^3, gamma ~= 3.18, lambda_g ~= 0.95
  For sigma-gate, sigma_c defaults to 270 Msun/pc^2 (≈ a0/(pi G)).

Usage example:
  python tools/run_sparc_grav_color.py --sparc_dir external_data/Rotmod_LTG --targets NGC3198 NGC2403 --plot
  python tools/run_sparc_grav_color.py --sparc_dir external_data/Rotmod_LTG --all --plot
  # Use gbar gate (RAR-like) or sigma gate
  python tools/run_sparc_grav_color.py --gate gbar --plot
  python tools/run_sparc_grav_color.py --gate sigma --sigma_c 270 --plot
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import json
import datetime as dt
import sys

# Ensure project root is on sys.path (so 'utils' can be imported when run from subdirs)
try:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except Exception:
    pass

KPC_M = 3.085677581e19  # meters in 1 kpc

# Use the robust SPARC loader that computes midplane densities
from utils.Utilities.sparc_io import load_single_sparc_galaxy, load_sparc_metadata
from models.nfw import v_model_nfw


def build_vbar_from_rotmod(gal: dict) -> np.ndarray:
    """Build baryonic circular velocity from SPARC rotmod components."""
    Vg = np.asarray(gal['V_gas_comp_kms'], dtype=float)
    Vd = np.asarray(gal['V_disk_comp_kms'], dtype=float)
    Vb = np.asarray(gal['V_bulge_comp_kms'], dtype=float)
    vbar_sq = np.clip(Vg**2 + Vd**2 + Vb**2, 0.0, None)
    return np.sqrt(vbar_sq)


def compute_grav_color_velocity(R_kpc: np.ndarray,
                                vbar_kms: np.ndarray,
                                rho_star_mid_base: np.ndarray,
                                rho_gas_mid: np.ndarray,
                                rho_c: float,
                                gamma_exp: float,
                                lambda_g: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute grav_color model velocities and xi(R) using midplane density.
    Returns (v_model, xi) arrays.
    """
    rho_mid = np.asarray(rho_star_mid_base, dtype=float) + np.asarray(rho_gas_mid, dtype=float)
    rho_c_safe = max(float(rho_c), 1e-30)
    # xi = 1 + lambda_g / (1 + (rho/rho_c)^gamma)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ratio_gamma = np.power(np.clip(rho_mid / rho_c_safe, 0.0, np.inf), float(gamma_exp))
        xi = 1.0 + float(lambda_g) / (1.0 + ratio_gamma)
    # Clip to physical range [1, 1+lambda_g]
    xi = np.clip(xi, 1.0, 1.0 + float(lambda_g))
    v_model = np.asarray(vbar_kms, dtype=float) * np.sqrt(np.maximum(xi, 0.0))
    return v_model, xi


def compute_rar_gate_velocity(R_kpc: np.ndarray,
                              vbar_kms: np.ndarray,
                              a0: float,
                              gamma_exp: float,
                              lambda_g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Acceleration-gated xi: xi = 1 + lambda_g / (1 + (gbar/a0)^gamma),
    with gbar = vbar^2 / R in SI units.
    Returns (v_model, xi, gbar_SI).
    """
    v_ms = np.asarray(vbar_kms, float) * 1e3
    R_m  = np.asarray(R_kpc,  float) * KPC_M
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        gbar = (v_ms * v_ms) / np.clip(R_m, 1.0, None)
        Sg = 1.0 / (1.0 + np.power(np.clip(gbar / float(a0), 0.0, np.inf), float(gamma_exp)))
        xi = 1.0 + float(lambda_g) * Sg
    xi = np.clip(xi, 1.0, 1.0 + float(lambda_g))
    v_model = np.asarray(vbar_kms, float) * np.sqrt(np.maximum(xi, 0.0))
    return v_model, xi, gbar


def compute_sigma_gate_velocity(vbar_kms: np.ndarray,
                                sigma_star_Msun_pc2: np.ndarray,
                                sigma_gas_Msun_pc2: np.ndarray,
                                sigma_c: float,
                                gamma_exp: float,
                                lambda_g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Surface-density gate: xi = 1 + lambda_g / (1 + (Sigma_b/sigma_c)^gamma),
    where Sigma_b = Sigma_star + Sigma_gas in Msun/pc^2.
    Returns (v_model, xi, Sigma_b).
    """
    Sigma_b = np.asarray(sigma_star_Msun_pc2, float) + np.asarray(sigma_gas_Msun_pc2, float)
    Sigma_b = np.clip(Sigma_b, 0.0, np.inf)
    sigma_c_safe = max(float(sigma_c), 1e-12)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ratio_gamma = np.power(np.clip(Sigma_b / sigma_c_safe, 0.0, np.inf), float(gamma_exp))
        xi = 1.0 + float(lambda_g) / (1.0 + ratio_gamma)
    xi = np.clip(xi, 1.0, 1.0 + float(lambda_g))
    v_model = np.asarray(vbar_kms, float) * np.sqrt(np.maximum(xi, 0.0))
    return v_model, xi, Sigma_b


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(r*r))) if len(r) else float('nan')


def chi2(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray, sigma_floor: float = 0.0) -> float:
    e = np.asarray(y_err, dtype=float)
    if sigma_floor and sigma_floor > 0:
        e = np.sqrt(e*e + float(sigma_floor)**2)
    e = np.where(e > 0, e, np.maximum(1.0, 0.05*np.maximum(np.asarray(y_true, float), 1.0)))
    r = (np.asarray(y_true, float) - np.asarray(y_pred, float)) / e
    return float(np.sum(r*r))


def evaluate_galaxy(galaxy_id: str,
                    sparc_dir: Path,
                    rho_c: float,
                    gamma_exp: float,
                    lambda_g: float,
                    model: str = "grav_color",
                    a0: float = 1.2e-10,
                    sigma_c: float = 270.0,
                    gate: str | None = None,
                    sigma_floor: float = 0.0,
                    assume_gas_hz_kpc: float = 0.1,
                    assume_stellar_hz_kpc: float = 0.3,
                    assume_hz_alpha: float = 0.0,
                    include_nfw: bool = True,
                    make_plot: bool = False,
                    plot_dir: Path | None = None,
                    preloaded: dict | None = None,
                    disable_mod: bool = False) -> dict | None:
    """Load a SPARC galaxy, compute GR and gated modified-gravity curves, and metrics.

    Parameters
    - model: legacy switch ('grav_color' or 'rar_gate'); kept for backward compatibility.
    - gate: optional explicit gate selector overriding 'model'. One of {'rho','gbar','sigma'}.
    - rho_c, a0, sigma_c: pivots for rho-, gbar-, and sigma-gates respectively.
    """
    if preloaded is not None:
        gal = preloaded
    else:
        gal = load_single_sparc_galaxy(
            galaxy_id,
            sparc_dir=sparc_dir,
            assume_gas_hz_kpc=assume_gas_hz_kpc,
            assume_stellar_hz_kpc=assume_stellar_hz_kpc,
            assume_hz_alpha=assume_hz_alpha,
        )
    if gal is None:
        return None

    R = np.asarray(gal['R_kpc'], float)
    v_obs = np.asarray(gal['V_obs'], float)
    e_v = np.asarray(gal['e_V_obs'], float)

    # Filter finite/positive
    mask = np.isfinite(R) & np.isfinite(v_obs) & np.isfinite(e_v) & (R > 0) & (v_obs > 0)
    R, v_obs, e_v = R[mask], v_obs[mask], e_v[mask]
    if len(R) == 0:
        return None

    vbar = build_vbar_from_rotmod(gal)[mask]

    # Densities at base M/L from loader
    rho_star_mid = np.asarray(gal['rho_star_mid_Msun_kpc3_baseML'], float)[mask]
    rho_gas_mid = np.asarray(gal['rho_gas_mid_Msun_kpc3'], float)[mask]

    # GR curve (baryons only)
    v_gr = vbar

    # Choose modified-gravity gate (gate overrides legacy 'model')
    v_mod = None
    xi = None
    if not disable_mod:
        selected = (gate or ("gbar" if model == "rar_gate" else "rho")).lower()
        if selected == "gbar":
            v_mod, xi = compute_rar_gate_velocity(R, vbar, a0=a0, gamma_exp=gamma_exp, lambda_g=lambda_g)[:2]
            model_name = "rar_gate"
        elif selected == "sigma":
            v_mod, xi = compute_sigma_gate_velocity(vbar, gal.get('Sigma_star_Msun_pc2_baseML')[mask],
                                                    gal.get('Sigma_gas_Msun_pc2')[mask],
                                                    sigma_c=sigma_c, gamma_exp=gamma_exp, lambda_g=lambda_g)[:2]
            model_name = "sigma_gate"
        else:
            v_mod, xi = compute_grav_color_velocity(R, vbar, rho_star_mid, rho_gas_mid,
                                                    rho_c=rho_c, gamma_exp=gamma_exp, lambda_g=lambda_g)
            model_name = "grav_color"
    else:
        model_name = "gr_only"

    # Optionally compute NFW baseline (fit per galaxy)
    nfw_metrics = None
    v_nfw_total = None
    if include_nfw:
        # Try to import scipy minimize; fallback to coarse grid if not available
        try:
            from scipy.optimize import minimize
        except Exception:
            minimize = None

        def _chi2_of(V200, c):
            vtot = v_model_nfw(R, vbar, V200, c)
            return chi2(v_obs, vtot, e_v, sigma_floor=sigma_floor)

        if minimize is not None:
            x0 = np.array([120.0, 10.0], dtype=float)
            def obj(x):
                V200, c = float(x[0]), float(x[1])
                # Boundaries
                if not (10.0 <= V200 <= 400.0 and 1.0 <= c <= 40.0):
                    return 1e12 + np.sum(x*x)
                return _chi2_of(V200, c)
            res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-3, "fatol": 1e-3})
            V200_best, c_best = float(res.x[0]), float(res.x[1])
        else:
            # Coarse grid fallback
            best = (np.inf, 120.0, 10.0)
            for V200 in np.linspace(40, 300, 66):
                for c in np.linspace(2, 25, 47):
                    c2v = _chi2_of(V200, c)
                    if c2v < best[0]:
                        best = (c2v, float(V200), float(c))
            V200_best, c_best = best[1], best[2]

        v_nfw_total = v_model_nfw(R, vbar, V200_best, c_best)
        chi2_nfw = chi2(v_obs, v_nfw_total, e_v, sigma_floor=sigma_floor)
        dof_nfw = max(len(R), 1)
        nfw_metrics = {
            'V200': V200_best,
            'c': c_best,
            'chi2': chi2_nfw,
            'chi2dof': chi2_nfw / dof_nfw,
            'rmse': rmse(v_obs, v_nfw_total),
        }

    # Metrics for GR and (optional) modified gravity
    chi2_gr = chi2(v_obs, v_gr, e_v, sigma_floor=sigma_floor)
    dof_gr  = max(len(R), 1)
    rmse_gr  = rmse(v_obs, v_gr)

    result = {
        'galaxy_id': galaxy_id,
        'n_points': int(len(R)),
        'model': model_name,
        'gate': (selected if not disable_mod else None),
        'rho_c': float(rho_c),
        'sigma_c': float(sigma_c),
        'gamma_exp': float(gamma_exp),
        'lambda_g': float(lambda_g),
        'a0': float(a0),
        'chi2_gr': chi2_gr,
        'chi2dof_gr': chi2_gr / dof_gr,
        'rmse_gr': rmse_gr,
    }
    if v_mod is not None:
        chi2_mod = chi2(v_obs, v_mod, e_v, sigma_floor=sigma_floor)
        dof_mod = max(len(R), 1)
        rmse_mod = rmse(v_obs, v_mod)
        result.update({
            'chi2_mod': chi2_mod,
            'chi2dof_mod': chi2_mod / dof_mod,
            'rmse_mod': rmse_mod,
        })
    if nfw_metrics is not None:
        result.update({
            'nfw_V200': nfw_metrics['V200'],
            'nfw_c': nfw_metrics['c'],
            'chi2_nfw': nfw_metrics['chi2'],
            'chi2dof_nfw': nfw_metrics['chi2dof'],
            'rmse_nfw': nfw_metrics['rmse'],
        })

    if make_plot and plot_dir is not None:
        import matplotlib.pyplot as plt
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})
        ax1.errorbar(R, v_obs, yerr=e_v, fmt='ko', ms=4, alpha=0.8, capsize=3, label='Observed')
        ax1.plot(R, v_gr, 'b--', lw=2, label='GR (baryons only)')
        if v_mod is not None:
            ax1.plot(R, v_mod, 'r-', lw=2.2, label=f'{model_name} (no DM)')
        if v_nfw_total is not None:
            ax1.plot(R, v_nfw_total, color='green', linestyle=':', lw=2.2, label='NFW (DM)')
        ax1.set_title(f"{galaxy_id}: GR vs {model_name}{' + NFW' if v_nfw_total is not None else ''}")
        ax1.set_ylabel('v (km/s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        ax2.axhline(0, color='k', lw=0.7)
        ax2.errorbar(R, v_obs - v_gr, yerr=e_v, fmt='b^', ms=4, alpha=0.7, capsize=2, label='GR res')
        if v_mod is not None:
            ax2.errorbar(R, v_obs - v_mod, yerr=e_v, fmt='ro', ms=4, alpha=0.7, capsize=2, label=f'{model_name} res')
        if v_nfw_total is not None:
            ax2.errorbar(R, v_obs - v_nfw_total, yerr=e_v, fmt='gs', ms=4, alpha=0.7, capsize=2, label='NFW res')
        ax2.set_xlabel('R (kpc)')
        ax2.set_ylabel('Residual (km/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=3 if v_nfw_total is not None else 2, loc='upper right')
        fig.tight_layout()
        out_file = plot_dir / f"{galaxy_id}_{model_name}.png"
        fig.savefig(out_file, dpi=140, bbox_inches='tight')
        plt.close(fig)
        result['plot'] = str(out_file)

    return result


def _standardize_id(gid: str) -> str:
    import re
    gid_std = gid.lower().replace(" ", "")
    gid_std = re.sub(r"([a-zA-Z]+)0+(\d+)", r"\1\2", gid_std)
    return gid_std


def _integrate_gas_mass_Msun(gal: dict) -> float:
    # Integrate M_gas ≈ 2π ∫ Σ_gas(R) R dR (Σ in Msun/pc^2, R in kpc)
    R = np.asarray(gal.get('R_kpc', []), float)
    Sigma = np.asarray(gal.get('Sigma_gas_Msun_pc2', []), float)
    if len(R) < 2 or len(Sigma) != len(R):
        return np.nan
    R_sorted_idx = np.argsort(R)
    R = R[R_sorted_idx]
    Sigma = Sigma[R_sorted_idx]
    dR = np.diff(R)
    R_mid = 0.5 * (R[:-1] + R[1:])
    # Msun/pc^2 * kpc^2 → Msun with 1e6 pc^2/kpc^2
    integrand = Sigma[:-1] * R_mid * dR * 1e6
    M = 2.0 * np.pi * np.nansum(integrand)
    return float(M)


def _meets_gold_density_gate(galaxy_id: str, sparc_dir: Path) -> tuple[bool, list[str]]:
    reasons = []
    # Load galaxy (use defaults)
    gal = load_single_sparc_galaxy(galaxy_id, sparc_dir=sparc_dir)
    if gal is None:
        return False, ["failed to load galaxy"]

    # Gas profile present: require real HIrad file
    if gal.get('hirad_path', None) is None:
        reasons.append("missing _HIrad.dat (no reconstruction allowed)")

    # Metadata for inclination, distance, Rd, flags
    df_meta = load_sparc_metadata(sparc_dir)
    row = None
    if df_meta is not None and 'Name' in df_meta.columns:
        std = _standardize_id(galaxy_id)
        df_meta['StdName'] = df_meta['Name'].apply(_standardize_id)
        m = df_meta[df_meta['StdName'] == std]
        if not m.empty:
            row = m.iloc[0]

    # Radial coverage & sampling: Rmax >= 4*Rd, Npts >= 25
    R = np.asarray(gal.get('R_kpc', []), float)
    if len(R) < 25:
        reasons.append("Npts < 25")
    Rmax = float(np.nanmax(R)) if len(R) else 0.0
    Rd_meta = float(row['Rdisk_kpc']) if (row is not None and 'Rdisk_kpc' in row and np.isfinite(row['Rdisk_kpc'])) else np.nan
    if np.isfinite(Rd_meta):
        if Rmax < 4.0 * Rd_meta:
            reasons.append(f"Rmax < 4 Rd (Rmax={Rmax:.2f}, Rd={Rd_meta:.2f})")
    else:
        reasons.append("missing Rdisk_kpc in metadata")

    # Inclination geometry: 45 <= i <= 80 with e_i <= 3 deg
    i_deg = float(gal.get('incl_deg', np.nan))
    ei_deg = float(gal.get('e_incl_deg', np.nan))
    if not (np.isfinite(i_deg) and 45.0 <= i_deg <= 80.0):
        reasons.append(f"inclination out of range (i={i_deg:.1f})")
    if not (np.isfinite(ei_deg) and ei_deg <= 3.0):
        reasons.append(f"inclination uncertainty too large (e_i={ei_deg:.1f})")

    # Distance quality: fractional error <= 10%
    D = float(gal.get('distance_Mpc', np.nan))
    eD = float(gal.get('e_distance_Mpc', np.nan))
    if not (np.isfinite(D) and np.isfinite(eD) and D > 0 and eD / D <= 0.10):
        reasons.append("distance fractional error > 10% or missing")

    # Dynamics quality: if 'Q' quality flag exists, require Q <= 2 (good)
    if row is not None and 'Q' in row and np.isfinite(row['Q']):
        try:
            qval = float(row['Q'])
            if qval > 2.0:
                reasons.append(f"quality flag Q={qval} (bars/warps risk)")
        except Exception:
            pass

    # Gas sanity if reconstructed
    pen = float(gal.get('gas_penalty_mass', 0.0))
    mismatch = float(gal.get('gas_mass_mismatch', 1.0))
    if pen > 2.0:
        reasons.append(f"gas mass penalty {pen:.2f} > 2")
    # Compare integrated Sigma_gas to M_HI
    M_HI = float(gal.get('M_HI_Msun', np.nan))
    M_int = _integrate_gas_mass_Msun(gal)
    if np.isfinite(M_HI) and np.isfinite(M_int) and M_HI > 0:
        ratio = M_int / M_HI
        if not (0.5 <= ratio <= 2.0):
            reasons.append(f"∫Σ_gas mismatch (ratio={ratio:.2f})")

    return (len(reasons) == 0, reasons)


def main():
    ap = argparse.ArgumentParser(description='Evaluate density/acceleration/surface-density gates on SPARC galaxies')
    ap.add_argument('--sparc_dir', type=str, default='external_data/Rotmod_LTG', help='Path to SPARC Rotmod_LTG directory')
    ap.add_argument('--targets', nargs='*', help='Specific galaxy IDs (e.g., NGC3198 NGC2403)')
    ap.add_argument('--all', action='store_true', help='Evaluate all galaxies in directory')
    # Gate parameters (same knobs, different pivots)
    ap.add_argument('--gate', choices=['rho','gbar','sigma'], default=None, help='Gate variable: rho (grav_color), gbar (RAR-like), or sigma (surface density)')
    ap.add_argument('--rho_c', type=float, default=3.48e8, help='rho_c in Msun/kpc^3 (rho gate)')
    ap.add_argument('--sigma_c', type=float, default=270.0, help='sigma_c in Msun/pc^2 (sigma gate)')
    ap.add_argument('--gamma_exp', type=float, default=3.18, help='gamma exponent')
    ap.add_argument('--lambda_g', type=float, default=0.95, help='lambda_g amplitude')
    # Legacy model switch for backward compatibility
    ap.add_argument('--model', choices=['grav_color','rar_gate'], default='grav_color',
                    help='Legacy: grav_color (rho gate) or rar_gate (gbar gate). Overridden by --gate if provided.')
    ap.add_argument('--a0', type=float, default=1.2e-10, help='RAR pivot acceleration in m/s^2 (gbar gate)')
    ap.add_argument('--sigma_floor', type=float, default=8.0, help='Velocity error floor (km/s)')
    ap.add_argument('--no_nfw', action='store_true', help='Disable NFW baseline fit/evaluation')
    ap.add_argument('--plot', action='store_true', help='Save per-galaxy plots')    ap.add_argument('--gr-only', action='store_true', help='Plot GR only (no modified-gravity curve)')
    ap.add_argument('--output_dir', type=str, default=None, help='Output directory')
    # Gold filter for density-gated models
    ap.add_argument('--no_gold_filter', action='store_true', help='Do not filter density-gated models by Gold criteria')
    ap.add_argument('--use_recommended_gold', action='store_true', help='Use a recommended Gold list of galaxies')
    args = ap.parse_args()

    sparc_dir = Path(args.sparc_dir)
    if not sparc_dir.exists():
        print(f"SPARC dir not found: {sparc_dir}")
        return 2

    # Determine galaxy list
    galaxies: list[str]
    if args.all:
        galaxies = [p.name.replace('_rotmod.dat', '') for p in sorted(sparc_dir.glob('*_rotmod.dat'))]
    elif args.targets:
        galaxies = args.targets
    else:
        # Default subset (high-quality examples)
        galaxies = ['NGC3198', 'NGC2403', 'NGC5055', 'NGC6946', 'NGC2841',
                    'DDO154', 'NGC6503', 'NGC7793', 'NGC2903', 'NGC7331']

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else Path(f'sparc_grav_color_results_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'

    # Optional override: use recommended Gold list
    if args.use_recommended_gold:
        recommended = ['NGC2403','NGC3198','NGC6503','NGC3621','NGC4258','DDO154','IC2574','UGC128','NGC3741']
        galaxies = [g for g in galaxies if any(_standardize_id(g)==_standardize_id(r) for r in recommended)]

    results = []
    for gid in galaxies:
        # If density-gated model, apply Gold filter unless disabled
        if args.model == 'grav_color' and not args.no_gold_filter:
            ok, reasons = _meets_gold_density_gate(gid, sparc_dir)
            if not ok:
                print(f"Skipping {gid} (fails Gold density checks): {', '.join(reasons)}")
                continue
            # Preload to avoid re-reading; but we need the preloaded object
            pre_gal = load_single_sparc_galaxy(gid, sparc_dir=sparc_dir)
        else:
            pre_gal = None

        print(f"Evaluating {gid}...")
        try:
            res = evaluate_galaxy(
                gid, sparc_dir,
                rho_c=args.rho_c,
                gamma_exp=args.gamma_exp,
                lambda_g=args.lambda_g,
                model=args.model,
                a0=args.a0,
                sigma_c=args.sigma_c,
                gate=args.gate,
                sigma_floor=args.sigma_floor,
                include_nfw=(not args.no_nfw),
                make_plot=args.plot,
                plot_dir=plot_dir if args.plot else None,
                preloaded=pre_gal,
                disable_mod=bool(args.gr_only),
            )
            if res is None:
                print(f"  ⚠️ Skipped {gid} (no usable data)")
                continue
            print(f"  → GR:  chi2/dof={res['chi2dof_gr']:.2f}, RMSE={res['rmse_gr']:.1f} km/s")
            print(f"  → {res['model'].upper()}:  chi2/dof={res['chi2dof_mod']:.2f}, RMSE={res['rmse_mod']:.1f} km/s")
            if 'chi2dof_nfw' in res:
                print(f"  → NFW: chi2/dof={res['chi2dof_nfw']:.2f}, RMSE={res['rmse_nfw']:.1f} km/s (V200={res['nfw_V200']:.1f}, c={res['nfw_c']:.1f})")
            results.append(res)
        except Exception as e:
            print(f"  ❌ Error in {gid}: {e}")

    # Write summary JSON and CSV
    if results:
        summary_json = out_dir / 'summary.json'
        with open(summary_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary to {summary_json}")

        # CSV
        import csv
        csv_path = out_dir / 'summary.csv'
        keys = ['galaxy_id','n_points','model','rho_c','gamma_exp','lambda_g','a0',
                'chi2_gr','chi2dof_gr','rmse_gr','chi2_mod','chi2dof_mod','rmse_mod',
                'nfw_V200','nfw_c','chi2_nfw','chi2dof_nfw','rmse_nfw']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                row = {k: r.get(k) for k in keys}
                w.writerow(row)
        print(f"Saved CSV to {csv_path}")

        # Basic aggregate diagnostics
        n_better = sum(1 for r in results if r['rmse_mod'] <= r['rmse_gr'])
        print(f"\nAggregate: {results[0]['model']} RMSE <= GR on {n_better}/{len(results)} galaxies (no DM)")
        if any('rmse_nfw' in r for r in results):
            n_better_vs_nfw = sum(1 for r in results if ('rmse_nfw' in r and r['rmse_mod'] <= r['rmse_nfw']))
            n_total_nfw = sum(1 for r in results if 'rmse_nfw' in r)
            print(f"Aggregate: {results[0]['model']} RMSE <= NFW on {n_better_vs_nfw}/{n_total_nfw} galaxies")

    else:
        print("No results to save.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

