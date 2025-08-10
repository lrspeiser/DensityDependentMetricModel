#!/usr/bin/env python3
"""
Compute ED-SPARC summary stats and a simple histogram figure for ΔlogZ(TFR−GR).
- Reads data/sparc_batch_summary.csv
- Writes:
  - docs/ed_sparc_stats.md (counts, medians, IQR)
  - images/ed_sparc_hist_dlogz.png (histogram)
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / 'data' / 'sparc_batch_summary.csv'
OUT_MD = REPO / 'docs' / 'ed_sparc_stats.md'
OUT_IMG = REPO / 'images' / 'ed_sparc_hist_dlogz.png'

def load_rows():
    rows = []
    if not CSV.exists():
        print(f"Missing {CSV}, run tools/generate_sparc_ed_table.py first.")
        return rows
    with open(CSV, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    rows = load_rows()
    if not rows:
        return
    dlogz = []
    for r in rows:
        v = to_float(r.get('dlogZ_ER_minus_GR'))
        if v is not None:
            dlogz.append(v)
    arr = np.array(dlogz, dtype=float)
    n = arr.size
    n_pos = int(np.sum(arr > 10.0))
    med = float(np.median(arr)) if n else float('nan')
    q25 = float(np.percentile(arr, 25)) if n else float('nan')
    q75 = float(np.percentile(arr, 75)) if n else float('nan')

    # Write markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# ED-SPARC summary stats (auto-generated)')
    lines.append('')
    lines.append(f'- Galaxies with ΔlogZ(TFR−GR) > 10: {n_pos}/{n}')
    lines.append(f'- Median ΔlogZ(TFR−GR): {med:.2f}')
    lines.append(f'- IQR ΔlogZ(TFR−GR): [{q25:.2f}, {q75:.2f}]')
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')

    # Histogram figure
    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    bins = np.linspace(min(-50.0, np.min(arr)), max(50.0, np.max(arr)), 30)
    plt.hist(arr, bins=bins, color='#377eb8', alpha=0.8)
    plt.axvline(10.0, color='red', ls='--', lw=1, label='ΔlogZ=10')
    plt.xlabel('ΔlogZ (TFR−GR)')
    plt.ylabel('Count')
    plt.title('ED-SPARC ΔlogZ(TFR−GR) histogram')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150)
    print(f"Wrote {OUT_MD} and {OUT_IMG}")

if __name__ == '__main__':
    main()

