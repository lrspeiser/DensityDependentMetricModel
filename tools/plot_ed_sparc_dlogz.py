#!/usr/bin/env python3
"""
Plot ΔlogZ across galaxies for ER/TFR vs GR, ER/TFR vs NFW, and Composite vs ER.
Outputs a PNG under docs/.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / 'ed_sparc_batch.csv'
OUT_PATH = REPO / 'docs' / 'ed_sparc_dlogz_summary.png'

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    names = []
    d_er_gr = []
    d_er_nfw = []
    d_comp_er = []
    for row in rows:
        names.append(row['galaxy'])
        z_gr = safe_float(row.get('logZ_GR'))
        z_nfw = safe_float(row.get('logZ_NFW'))
        z_er = safe_float(row.get('logZ_ER'))
        z_comp = safe_float(row.get('logZ_COMP'))
        d_er_gr.append((z_er - z_gr) if (z_er is not None and z_gr is not None) else np.nan)
        d_er_nfw.append((z_er - z_nfw) if (z_er is not None and z_nfw is not None) else np.nan)
        d_comp_er.append((z_comp - z_er) if (z_comp is not None and z_er is not None) else np.nan)

    idx = np.arange(len(names))
    width = 0.28

    plt.figure(figsize=(max(8, len(names)*0.5), 6))
    plt.axhline(0, color='k', lw=1, alpha=0.5)
    plt.bar(idx - width, d_er_gr, width=width, label='TFR−GR', color='#d95f02')
    plt.bar(idx, d_er_nfw, width=width, label='TFR−NFW', color='#1b9e77')
    plt.bar(idx + width, d_comp_er, width=width, label='Composite−TFR', color='#7570b3')
    plt.xticks(idx, names, rotation=75, ha='right')
    plt.ylabel('ΔlogZ')
    plt.title('SPARC Batch: Evidence Differences')
    plt.legend(frameon=False)
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")

if __name__ == '__main__':
    main()

