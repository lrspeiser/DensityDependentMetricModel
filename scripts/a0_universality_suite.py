#!/usr/bin/env python3
"""
a0_universality_suite.py

Implements the reviewer-requested analyses around a0 universality and the
environment term, using artifacts already produced by next_steps_from_run.py.

Modes
- universal: assume strict universal a0 across the SPARC subset; find the global
  minimizer of total χ² by summing per-galaxy grids at common ln a0; report
  best ln a0, total χ², DOF and reduced χ², and write a CSV + PNG.
- hierarchical: summarize hierarchical ln a0 scatter (σ) from results_root
  (reads hierarchical_a0_posterior_summary.json or MLE fallback), and write a
  compact JSON summary.
- ablation: compare two results roots (env off vs on) and report the marginal
  utility of the environment term via Δχ²_total, ΔAIC, ΔBIC approximations.
- corr: inspect residual correlations by correlating per-galaxy reduced χ² with
  SPARC metadata (surface brightness, inclination, gas fraction proxies).

Inputs (common)
- --results-root: results/next_steps/<run_name> with sparc_a0_grids/*.csv and
  sparc_a0_summary.csv (and optional hierarchical_* files from orchestration).
- Some modes also need --results-root2 (ablation) and --sparc-dir (corr).

Outputs
- universal: writes universal_a0_summary.json and universal_a0_curve.png.
- hierarchical: writes hierarchical_a0_quick_summary.json.
- ablation: writes env_ablation_summary.json.
- corr: writes residual_correlations.json and residual_correlations.csv.

Note
- This suite reads precomputed grids; it does not re-fit galaxies.
- Environment term ablation treats the two runs as alternative models over the
  same sample and uses simple information criteria approximations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import math


def _read_grid_dir(grids_dir: Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    rows: list[tuple[np.ndarray, np.ndarray, str]] = []
    for p in sorted(grids_dir.glob('*.csv')):
        try:
            ln_a = []
            chi2 = []
            with p.open('r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    ln_a.append(float(parts[0]))
                    chi2.append(float(parts[1]))
            if len(ln_a) >= 4:
                rows.append((np.asarray(ln_a, float), np.asarray(chi2, float), p.stem))
        except Exception:
            continue
    return rows


def _read_summary_csv(path: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    if not path.exists():
        return m
    with path.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                a0_best = float(parts[1]) if parts[1].lower() != 'nan' else float('nan')
            except Exception:
                a0_best = float('nan')
            try:
                chi2_rar = float(parts[2])
                chi2_gr = float(parts[3])
            except Exception:
                chi2_rar = float('nan'); chi2_gr = float('nan')
            try:
                dof = int(parts[4])
            except Exception:
                dof = None
            m[name] = {'a0_best': a0_best, 'chi2_rar': chi2_rar, 'chi2_gr': chi2_gr, 'dof': dof}
    return m


def mode_universal(results_root: Path) -> None:
    grids_dir = results_root / 'sparc_a0_grids'
    rows = _read_grid_dir(grids_dir)
    if not rows:
        raise SystemExit(f'No grids found under {grids_dir}')
    # Determine overlapping ln a0 range
    lo = max(float(np.nanmin(x)) for x, _, _ in rows)
    hi = min(float(np.nanmax(x)) for x, _, _ in rows)
    if hi <= lo:
        raise SystemExit('No overlapping ln a0 range across SPARC grids')
    ln_a0 = np.linspace(lo, hi, 320)
    # Sum chi2 over galaxies (interpolate where needed)
    chi2_tot = np.zeros_like(ln_a0)
    valid_counts = np.zeros_like(ln_a0)
    for xa, c2, _ in rows:
        c2i = np.interp(ln_a0, xa, c2, left=np.nan, right=np.nan)
        m = np.isfinite(c2i)
        chi2_tot[m] += c2i[m]
        valid_counts[m] += 1
    # Require a minimum coverage fraction (e.g., 80% galaxies) at a grid point
    min_cov = 0.8 * len(rows)
    m_cov = valid_counts >= max(1, int(min_cov))
    if not np.any(m_cov):
        # fallback to any coverage
        m_cov = valid_counts >= 1
    # Find best ln a0 among covered points
    idx = int(np.nanargmin(np.where(m_cov, chi2_tot, np.nan)))
    ln_a0_best = float(ln_a0[idx]); a0_best = 10.0 ** ln_a0_best
    chi2_best = float(chi2_tot[idx])
    # DOF from summary CSV if available
    smap = _read_summary_csv(results_root / 'sparc_a0_summary.csv')
    dof_tot = int(np.nansum([v.get('dof') for v in smap.values() if v.get('dof') is not None])) if smap else None
    red_chi2 = (chi2_best / dof_tot) if (dof_tot and dof_tot > 0) else None
    out = {
        'ln_a0_best': ln_a0_best,
        'a0_best_SI': a0_best,
        'chi2_total': chi2_best,
        'dof_total': dof_tot,
        'reduced_chi2': red_chi2,
        'grid_points_used': int(np.sum(m_cov)),
        'N_galaxies': len(rows),
        'results_root': str(results_root),
    }
    (results_root / 'universal_a0_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    # Plot chi2 curve
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(results_root, exist_ok=True)
        plt.figure(figsize=(6.4,4.1))
        plt.plot(ln_a0, chi2_tot, lw=1.6, color='#1f2937')
        plt.axvline(ln_a0_best, color='#ef4444', ls='--', lw=1.2, label=f'best ln a0={ln_a0_best:.3f}')
        plt.xlabel('ln a0 (natural log)'); plt.ylabel('Σ χ² over galaxies')
        plt.title('Universal a0 fit (strict)')
        plt.legend(frameon=False)
        plt.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(results_root / 'universal_a0_curve.png', dpi=140); plt.close()
    except Exception:
        pass
    print(json.dumps(out, indent=2))


def mode_hierarchical(results_root: Path) -> None:
    # Prefer Bayesian posterior summary if present
    post = results_root / 'hierarchical_a0_posterior_summary.json'
    mle = results_root / 'hierarchical_a0_summary.json'
    out = {'results_root': str(results_root)}
    if post.exists():
        j = json.loads(post.read_text(encoding='utf-8'))
        out.update({
            'mu_ln_a0_p50': j.get('mu_ln_a0', {}).get('p50'),
            'sigma_ln_a0_p50': j.get('sigma_ln_a0', {}).get('p50'),
            'n_gal': j.get('n_gal'),
            'source': 'posterior',
        })
    elif mle.exists():
        j = json.loads(mle.read_text(encoding='utf-8'))
        out.update({
            'mu_ln_a0_mle': j.get('mu'),
            'sigma_ln_a0_mle': j.get('sigma'),
            'n_gal': j.get('n_gal'),
            'source': 'mle_grid',
        })
    else:
        raise SystemExit('No hierarchical summaries found; run next_steps_from_run with hierarchical flags')
    (results_root / 'hierarchical_a0_quick_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


def _sum_total_chi2(results_root: Path) -> tuple[float, int | None]:
    smap = _read_summary_csv(results_root / 'sparc_a0_summary.csv')
    chi2 = float(np.nansum([v.get('chi2_rar') for v in smap.values()])) if smap else float('nan')
    dof = int(np.nansum([v.get('dof') for v in smap.values() if v.get('dof') is not None])) if smap else None
    return chi2, dof


def mode_ablation(results_root_off: Path, results_root_on: Path) -> None:
    # Compare total chi2 and IC across two runs (env off vs on)
    chi2_off, dof_off = _sum_total_chi2(results_root_off)
    chi2_on, dof_on = _sum_total_chi2(results_root_on)
    # Use same sample size assumption; approximate N_eff = dof_off (fallback dof_on)
    N_eff = dof_off or dof_on or 0
    # Parameter counts: off: k_off ~ 1 (universal a0); on: k_on ~ 2 (a0 + ζ_env) if both were global; if per-galaxy a0 grids used, this is a rough proxy
    k_off = 1; k_on = 2
    # AIC = chi2 + 2k; BIC = chi2 + k ln N
    aic_off = chi2_off + 2*k_off
    aic_on  = chi2_on  + 2*k_on
    bic_off = chi2_off + (k_off * math.log(max(N_eff, 1)))
    bic_on  = chi2_on  + (k_on  * math.log(max(N_eff, 1)))
    out = {
        'results_off': str(results_root_off),
        'results_on': str(results_root_on),
        'chi2_total_off': chi2_off,
        'chi2_total_on': chi2_on,
        'delta_chi2_on_minus_off': chi2_on - chi2_off,
        'AIC_off': aic_off, 'AIC_on': aic_on, 'delta_AIC': aic_on - aic_off,
        'BIC_off': bic_off, 'BIC_on': bic_on, 'delta_BIC': bic_on - bic_off,
        'k_off': k_off, 'k_on': k_on, 'N_eff': N_eff,
        'note': 'IC approximations assume global (a0, ζ_env) only. For per-galaxy a0, treat as model class comparison via Δχ² or use hierarchical evidence.',
    }
    (results_root_on / 'env_ablation_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


def _load_sparc_meta(sparc_dir: Path):
    import pandas as pd
    ms = Path(sparc_dir) / 'MasterSheet_SPARC.csv'
    if not ms.exists():
        return None
    try:
        df = pd.read_csv(ms)
        return df
    except Exception:
        return None


def mode_corr(results_root: Path, sparc_dir: Path) -> None:
    # Correlate per-galaxy reduced chi2 with surface brightness, inclination, gas fraction proxy
    try:
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr
    except Exception as e:
        raise SystemExit(f'Correlation analysis requires pandas and scipy ({e})')
    smap = _read_summary_csv(results_root / 'sparc_a0_summary.csv')
    if not smap:
        raise SystemExit('sparc_a0_summary.csv not found or empty')
    df_meta = _load_sparc_meta(Path(sparc_dir))
    if df_meta is None:
        raise SystemExit('SPARC MasterSheet_SPARC.csv not found; provide --sparc-dir pointing to Rotmod_LTG parent with MasterSheet')
    rows = []
    for gal, vals in smap.items():
        red = None
        if vals.get('dof') and vals['dof'] > 0 and math.isfinite(vals.get('chi2_rar', float('nan'))):
            red = vals['chi2_rar'] / vals['dof']
        if red is None or not math.isfinite(red):
            continue
        # Join meta by galaxy name (allow loose match)
        try:
            hit = df_meta[df_meta['Name'].astype(str).str.lower().str.replace(' ', '') == gal.lower().replace(' ', '')]
            if hit.empty:
                # Attempt no-leading-zero match for NGC0xxx
                import re
                gal_std = re.sub(r'([A-Za-z]+)0+(\d+)', r'\1\2', gal)
                hit = df_meta[df_meta['Name'].astype(str).str.lower().str.replace(' ', '') == gal_std.lower()]
            if hit.empty:
                continue
            row = hit.iloc[0]
            # Features: SBdisk0 (central SB), Inc, gas fraction proxy = MHI / L_3p6
            SB0 = float(row.get('SBdisk0_Lsun_pc2', np.nan))
            Inc = float(row.get('Inc', np.nan))
            L36 = float(row.get('L_3p6_1e9Lsun', np.nan)) * 1e9 if pd.notnull(row.get('L_3p6_1e9Lsun')) else np.nan
            MHI = float(row.get('MHI_1e9Msun', np.nan)) * 1e9 if pd.notnull(row.get('MHI_1e9Msun')) else np.nan
            gas_frac = (MHI / L36) if (math.isfinite(MHI) and math.isfinite(L36) and L36>0) else np.nan
            rows.append({'galaxy': gal, 'red_chi2': red, 'SB0_Lsun_pc2': SB0, 'Inc_deg': Inc, 'gas_frac_MHI_over_L36': gas_frac})
        except Exception:
            continue
    if not rows:
        raise SystemExit('No rows to correlate; check metadata join')
    df = pd.DataFrame(rows)
    # Compute correlations (Pearson and Spearman) for each feature vs red_chi2
    feats = ['SB0_Lsun_pc2', 'Inc_deg', 'gas_frac_MHI_over_L36']
    corr = {}
    for f in feats:
        s = df[['red_chi2', f]].dropna()
        if len(s) < 5:
            continue
        try:
            pr = pearsonr(s['red_chi2'], s[f])
            sr = spearmanr(s['red_chi2'], s[f])
            corr[f] = {
                'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
                'spearman_rho': float(sr.statistic), 'spearman_p': float(sr.pvalue),
                'N': int(len(s)),
            }
        except Exception:
            continue
    (results_root / 'residual_correlations.json').write_text(json.dumps(corr, indent=2), encoding='utf-8')
    (results_root / 'residual_correlations.csv').write_text('feature,pearson_r,pearson_p,spearman_rho,spearman_p,N\n' + '\n'.join(
        f"{k},{v['pearson_r']:.6f},{v['pearson_p']:.3g},{v['spearman_rho']:.6f},{v['spearman_p']:.3g},{v['N']}" for k,v in corr.items()
    ), encoding='utf-8')
    print(json.dumps(corr, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['universal', 'hierarchical', 'ablation', 'corr'])
    ap.add_argument('--results-root', required=True, help='results/next_steps/<run_name> directory')
    ap.add_argument('--results-root2', default=None, help='Second results dir (for ablation env on/off)')
    ap.add_argument('--sparc-dir', default=None, help='SPARC parent directory containing MasterSheet_SPARC.csv (for corr)')
    args = ap.parse_args()

    rr = Path(args.results_root)
    if args.mode == 'universal':
        mode_universal(rr)
    elif args.mode == 'hierarchical':
        mode_hierarchical(rr)
    elif args.mode == 'ablation':
        if not args.results_root2:
            raise SystemExit('--results-root2 is required for ablation mode')
        mode_ablation(rr, Path(args.results_root2))
    elif args.mode == 'corr':
        if not args.sparc_dir:
            raise SystemExit('--sparc-dir is required for corr mode')
        mode_corr(rr, Path(args.sparc_dir))


if __name__ == '__main__':
    main()
