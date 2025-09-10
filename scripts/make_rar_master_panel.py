#!/usr/bin/env python3
"""
make_rar_master_panel.py

Build a RAR master panel:
- SPARC scatter: (log10 g_bar, log10 g_obs) across current selection
- DGG posterior band: median and 16–84% envelope from hierarchical ln a0 posterior
- Optional ΛCDM band overlay via CSV (columns: log10_gbar, log10_gobs_lo, log10_gobs_hi)

Inputs
- --sparc-dir: SPARC Rotmod_LTG path
- --results-root: results/next_steps/<run>
- --images-root: images/next_steps/<run> (optional)
- --sample-csv: selection CSV to list galaxies (defaults to results_root/sparc_a0_summary.csv)
- --lcdm-band: optional CSV to overlay an LCDM band with columns described above
- --out: optional explicit output PNG path (defaults under images_root)

Outputs
- images_root/rar_master_panel.png
- results_root/rar_master_panel_source.csv (scatter points used)

Notes
- This script uses base M/L components from SPARC to compute V_bar and then
  g_bar = (V_bar^2 / R) * K. g_obs computed similarly from V_obs.
- DGG band uses g_eff = D * g_bar with D = 0.5 + sqrt(0.25 + a0/g_bar) and a0 ~ LogNormal(μ,σ) in ln space.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import math
import csv

ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14


def _import_sparc_loader(repo_root: Path):
    # load utils/Utilities/sparc_io.py without relying on installed package
    import importlib.util
    p = repo_root / 'utils' / 'Utilities' / 'sparc_io.py'
    if not p.exists():
        raise FileNotFoundError('utils/Utilities/sparc_io.py not found')
    spec = importlib.util.spec_from_file_location('sparc_io_runtime_rar', str(p))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, 'load_single_sparc_galaxy')


def _std_id(gid: str) -> str:
    import re
    gid_std = gid.lower().replace(' ', '')
    gid_std = re.sub(r'([a-zA-Z]+)0+(\d+)', r'\1\2', gid_std)
    return gid_std


def _load_selection(sample_csv: Path) -> list[str]:
    glx: list[str] = []
    with sample_csv.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) == 0:
                continue
            glx.append(parts[0])
    return glx


essential_cols = ['R_kpc', 'V_obs', 'e_V_obs', 'V_gas_comp_kms', 'V_disk_comp_kms', 'V_bulge_comp_kms']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc-dir', required=True)
    ap.add_argument('--results-root', required=True)
    ap.add_argument('--images-root', default=None)
    ap.add_argument('--sample-csv', default=None)
    ap.add_argument('--lcdm-band', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-a0-samples', type=int, default=2000)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    images_root = Path(args.images_root) if args.images_root else (results_root.parents[1] / 'images' / 'next_steps' / results_root.name)
    sample_csv = Path(args.sample_csv) if args.sample_csv else (results_root / 'sparc_a0_summary.csv')

    repo_root = Path.cwd()
    load_single = _import_sparc_loader(repo_root)

    # Load hierarchical posterior summary
    post_json = results_root / 'hierarchical_a0_posterior_summary.json'
    if not post_json.exists():
        raise FileNotFoundError('hierarchical_a0_posterior_summary.json not found; run hierarchical step first')
    pj = json.loads(post_json.read_text(encoding='utf-8'))
    # Posterior summary naming historically used "ln" but values are often in log10.
    # Prefer explicit keys if available; otherwise, detect base by magnitude.
    mu = None; sigma = None
    # Explicit log10 keys
    if 'mu_log10_a0' in pj and 'sigma_log10_a0' in pj:
        mu = float(pj['mu_log10_a0']['p50'])
        sigma = float(pj['sigma_log10_a0']['p50'])
        log_base = 10
    elif 'mu_ln_a0' in pj and 'sigma_ln_a0' in pj:
        mu = float(pj['mu_ln_a0']['p50'])
        sigma = float(pj['sigma_ln_a0']['p50'])
        # Heuristic: if |mu| ~ 8–13 it's almost certainly log10(a0 in SI); if |mu| ~ 20–25 it's ln.
        log_base = 10 if (-13.5 < mu < -8.0) else 'e'
    else:
        raise KeyError('Hierarchical a0 summary missing expected keys (mu_*, sigma_*)')

    galaxies = _load_selection(sample_csv)

    # Build scatter arrays
    xs = []  # log10 g_bar
    ys = []  # log10 g_obs
    used = 0
    for gid in galaxies:
        try:
            data = load_single(gid, sparc_dir=str(Path(args.sparc_dir)))
            if not data:
                continue
            R = np.asarray(data['R_kpc'], float)
            Vobs = np.asarray(data['V_obs'], float)
            Vgas = np.asarray(data['V_gas_comp_kms'], float)
            Vdisk = np.asarray(data['V_disk_comp_kms'], float)
            Vbul = np.asarray(data['V_bulge_comp_kms'], float)
            if min(map(len, [R, Vobs, Vgas, Vdisk, Vbul])) < 3:
                continue
            Vbar = np.sqrt(np.maximum(Vgas,0.0)**2 + np.maximum(Vdisk,0.0)**2 + np.maximum(Vbul,0.0)**2)
            gbar = ACC_M_S2_PER_KMS2_PER_KPC * np.maximum(Vbar,0.0)**2 / np.maximum(R, 1e-12)
            gobs = ACC_M_S2_PER_KMS2_PER_KPC * np.maximum(Vobs,0.0)**2 / np.maximum(R, 1e-12)
            m = np.isfinite(gbar) & np.isfinite(gobs) & (gbar>0) & (gobs>0)
            if np.any(m):
                xs.extend(np.log10(gbar[m]))
                ys.extend(np.log10(gobs[m]))
                used += 1
        except Exception:
            continue

    if used == 0:
        raise RuntimeError('No SPARC points found for selection; check SPARC dir and CSV contents')

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    # Save source data
    src_csv = results_root / 'rar_master_panel_source.csv'
    with src_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['log10_gbar', 'log10_gobs'])
        for a, b in zip(xs, ys):
            w.writerow([f'{a:.9f}', f'{b:.9f}'])

    # Prepare DGG posterior band
    rng = np.random.default_rng(0)
    n = max(int(args.n_a0_samples), 200)
    if log_base == 10:
        a0_samp = 10 ** (rng.normal(mu, sigma, size=n))  # log10 a0 ~ N(mu,sigma)
    else:
        a0_samp = np.exp(rng.normal(mu, sigma, size=n))  # ln a0 ~ N(mu,sigma)
    # gbar grid
    gmin = max(1e-13, 10**(np.nanmin(xs) - 2.0))
    gmax = min(1e-8, 10**(np.nanmax(xs) + 0.5))
    ggrid = np.logspace(np.log10(gmin), np.log10(gmax), 400)
    # Compute D = 0.5+sqrt(0.25 + a0/gbar); g_eff = D*gbar
    band = []
    for gb in ggrid:
        D = 0.5 + np.sqrt(0.25 + np.maximum(a0_samp, 0.0) / max(gb, 1e-30))
        geff = D * gb
        band.append([
            float(np.nanpercentile(np.log10(geff), 16)),
            float(np.nanpercentile(np.log10(geff), 50)),
            float(np.nanpercentile(np.log10(geff), 84)),
        ])
    band = np.asarray(band, float)

    # Plot
    import matplotlib.pyplot as plt
    images_root.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out) if args.out else (images_root / 'rar_master_panel.png')

    plt.figure(figsize=(7.4, 5.8))
    plt.scatter(xs, ys, s=6, alpha=0.35, color='black', label='SPARC (this selection)')
    # DGG band
    xg = np.log10(ggrid)
    plt.plot(xg, band[:,1], color='#e11d48', lw=2.0, label='DGG median (hierarchical)')
    plt.fill_between(xg, band[:,0], band[:,2], color='#fda4af', alpha=0.35, label='DGG 16–84%')

    # Optional LCDM band
    if args.lcdm_band:
        bp = Path(args.lcdm_band)
        try:
            xs_b = []
            lo_b = []
            hi_b = []
            with bp.open('r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
                for ln in f:
                    if not ln.strip():
                        continue
                    a,b,c = [s.strip() for s in ln.strip().split(',')[:3]]
                    xs_b.append(float(a))
                    lo_b.append(float(b))
                    hi_b.append(float(c))
            xs_b = np.asarray(xs_b, float)
            lo_b = np.asarray(lo_b, float)
            hi_b = np.asarray(hi_b, float)
            # Draw shaded band
            ii = np.argsort(xs_b)
            plt.fill_between(xs_b[ii], lo_b[ii], hi_b[ii], color='#60a5fa', alpha=0.25, label='ΛCDM band (input)')
        except Exception:
            pass

    plt.xlabel('log10 g_bar [m s^-2]')
    plt.ylabel('log10 g_obs [m s^-2]')
    plt.title('RAR master panel — SPARC selection with DGG posterior band')
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    print(f'Saved: {out_png} and {src_csv}')


if __name__ == '__main__':
    main()

