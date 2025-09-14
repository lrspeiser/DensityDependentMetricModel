#!/usr/bin/env python3
# Compare model curves (AG/DG/hybrid) to SPARC observed velocities and compute RMS/coverage.

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_obs(obs_csv: Path, galaxies: list[str]):
    data = {}
    with obs_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gid = row['galaxy_id']
            if gid not in galaxies: continue
            data.setdefault(gid, {'R': [], 'Vobs': [], 'err': []})
            data[gid]['R'].append(float(row['R_kpc']))
            data[gid]['Vobs'].append(float(row['Vobs_kms']))
            data[gid]['err'].append(float(row['err_kms']))
    for gid in list(data.keys()):
        for k in data[gid]:
            data[gid][k] = np.array(data[gid][k])
    return data


def load_model(model_csv: Path):
    R, Vmod = [], []
    with model_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            R.append(float(row['R_kpc']))
            Vmod.append(float(row['V_model_kms']))
    return np.array(R), np.array(Vmod)


def metrics_one(gid: str, obs: dict, model_csv: Path, out_plot_dir: Path | None = None) -> dict:
    Rm, Vm = load_model(model_csv)
    Ro, Vo, eo = obs['R'], obs['Vobs'], obs['err']
    # Interpolate model at obs radii
    Vm_at_R = np.interp(Ro, Rm, Vm, left=Vm[0], right=Vm[-1])
    resid = Vo - Vm_at_R
    rms = float(np.sqrt(np.mean(resid**2)))
    frac_10 = float(np.mean(np.abs(resid) <= 10.0))  # within 10 km/s
    frac_20 = float(np.mean(np.abs(resid) <= 20.0))

    if out_plot_dir is not None:
        out_plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=140)
        ax.errorbar(Ro, Vo, yerr=eo, fmt='o', ms=3.5, lw=0.7, color='k', alpha=0.8, label='Observed')
        ax.plot(Rm, Vm, '-', lw=1.6, color='tab:red', label='Model')
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('V [km/s]')
        ax.set_title(f'{gid}: RMS={rms:.1f} km/s')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_plot_dir / f'{gid}_rc_compare.png')
        plt.close(fig)
    return {
        'galaxy': gid,
        'n': int(len(Ro)),
        'rms_kms': rms,
        'coverage_10kms': frac_10,
        'coverage_20kms': frac_20,
    }


def main():
    ap = argparse.ArgumentParser(description='SPARC model-vs-observed metrics')
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--plots-dir', default='')
    args = ap.parse_args()

    obs = load_obs(Path(args.obs_csv), args.galaxies)
    rows = []
    plot_dir = Path(args.plots_dir) if args.plots_dir else None
    for gid in args.galaxies:
        model_csv = Path(args.models_dir) / f"{gid}_gate_model.csv"
        if not model_csv.exists():
            continue
        rows.append(metrics_one(gid, obs[gid], model_csv, plot_dir))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['galaxy','n','rms_kms','coverage_10kms','coverage_20kms'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote metrics to', args.out_csv)


if __name__ == '__main__':
    main()
