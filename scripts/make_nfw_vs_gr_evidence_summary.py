#!/usr/bin/env python3
"""
make_nfw_vs_gr_evidence_summary.py

Aggregate ΔlogZ(NFW−GR) over the current SPARC selection by reading
per-galaxy JSON evidence files under images/.

Inputs
- --results-root: results/next_steps/<run>
- --images-dir: images/ directory (defaults to repo images/)
- --sample-csv: selection CSV; defaults to results_root/sparc_a0_summary.csv

Outputs
- results_root/delta_logZ_nfw_vs_gr.csv
- images_root/delta_logZ_nfw_vs_gr.png (histogram)
- results_root/delta_logZ_nfw_vs_gr_summary.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import math


def _std_id(gid: str) -> str:
    import re
    gid_std = gid.lower().replace(' ', '')
    gid_std = re.sub(r'([a-zA-Z]+)0+(\d+)', r'\1\2', gid_std)
    return gid_std


def _read_selection(csv_path: Path) -> list[str]:
    out: list[str] = []
    with csv_path.open('r', encoding='utf-8') as f:
        f.readline()
        for ln in f:
            if not ln.strip():
                continue
            out.append(ln.strip().split(',')[0])
    return out


def _read_logZ(json_path: Path) -> float | None:
    try:
        j = json.loads(json_path.read_text(encoding='utf-8'))
        ev = j.get('evidence', {})
        val = ev.get('logZ')
        return float(val) if val is not None else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', required=True)
    ap.add_argument('--images-dir', default='images')
    ap.add_argument('--sample-csv', default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    images_root = results_root.parents[1] / 'images' / 'next_steps' / results_root.name
    images_dir = Path(args.images_dir)
    sample_csv = Path(args.sample_csv) if args.sample_csv else (results_root / 'sparc_a0_summary.csv')

    galaxies = _read_selection(sample_csv)

    rows = []
    deltas = []
    for gid in galaxies:
        std = _std_id(gid)
        p_gr = images_dir / f'sparc_gr_evidence_{std}.json'
        p_nfw = images_dir / f'sparc_nfw_evidence_{std}.json'
        if (not p_gr.exists()) or (not p_nfw.exists()):
            # try repo-level images/ fallback
            p_gr2 = Path('images') / f'sparc_gr_evidence_{std}.json'
            p_nfw2 = Path('images') / f'sparc_nfw_evidence_{std}.json'
            if p_gr2.exists():
                p_gr = p_gr2
            if p_nfw2.exists():
                p_nfw = p_nfw2
        lg = _read_logZ(p_gr)
        lnfw = _read_logZ(p_nfw)
        if (lg is None) or (lnfw is None):
            continue
        d = float(lnfw - lg)
        rows.append((gid, lnfw, lg, d))
        deltas.append(d)

    # Write CSV
    out_csv = results_root / 'delta_logZ_nfw_vs_gr.csv'
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('galaxy,logZ_NFW,logZ_GR,delta_logZ_NFW_vs_GR\n')
        for gid, lnfw, lg, d in rows:
            f.write(f'{gid},{lnfw:.6f},{lg:.6f},{d:.6f}\n')

    # Summary JSON
    deltas = np.asarray(deltas, float)
    summary = {
        'N': int(len(deltas)),
        'mean': float(np.nanmean(deltas)) if len(deltas) else float('nan'),
        'median': float(np.nanmedian(deltas)) if len(deltas) else float('nan'),
        'p16': float(np.nanpercentile(deltas, 16)) if len(deltas) else float('nan'),
        'p84': float(np.nanpercentile(deltas, 84)) if len(deltas) else float('nan'),
    }
    (results_root / 'delta_logZ_nfw_vs_gr_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Histogram
    try:
        import matplotlib.pyplot as plt
        images_root.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.0, 4.6))
        nb = min(50, max(10, int(np.sqrt(max(len(deltas), 1)))))
        plt.hist(deltas, bins=nb, color='#10b981', alpha=0.85, edgecolor='white')
        plt.axvline(0.0, color='k', ls=':', lw=1.2)
        plt.xlabel('Δlog Z (NFW − GR)')
        plt.ylabel('Number of galaxies')
        plt.title(f'NFW vs GR evidence across sample (N={len(deltas)})')
        plt.grid(alpha=0.2)
        out_png = images_root / 'delta_logZ_nfw_vs_gr.png'
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    except Exception:
        pass

    print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()

