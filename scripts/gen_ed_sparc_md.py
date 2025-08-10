#!/usr/bin/env python3
"""
Generate a Markdown Extended Data table (ED-SPARC) from the aggregated CSV
images/sparc_evidence_triplet_summary.csv and write to docs/ED-SPARC.md.
"""
from __future__ import annotations
import csv
from pathlib import Path

def main():
    repo = Path(__file__).resolve().parents[1]
    csv_path = repo / 'images' / 'sparc_evidence_triplet_summary.csv'
    out_md = repo / 'docs' / 'ED-SPARC.md'

    if not csv_path.exists():
        raise FileNotFoundError(f'Missing CSV: {csv_path}')

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Sort by galaxy name for consistency
    rows.sort(key=lambda r: (r.get('galaxy') or '').lower())

    headers = [
        'Galaxy', 'logZ_GR', 'logZ_NFW', 'logZ_ER',
        'ΔlogZ(ER−GR)', 'ΔlogZ(ER−NFW)'
    ]

    def fmt(x):
        if x is None or x == '' or x == 'None':
            return ''
        try:
            val = float(x)
        except Exception:
            return str(x)
        return f'{val:.3f}'

    # Build Markdown table
    lines = []
    lines.append('# Extended Data: SPARC evidence triplet (ER, GR, NFW)')
    lines.append('')
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

    for r in rows:
        line = [
            r.get('galaxy',''),
            fmt(r.get('logZ_GR')),
            fmt(r.get('logZ_NFW')),
            fmt(r.get('logZ_ER')),
            fmt(r.get('Delta_ER_minus_GR')),
            fmt(r.get('Delta_ER_minus_NFW')),
        ]
        lines.append('| ' + ' | '.join(line) + ' |')

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote: {out_md}')

if __name__ == '__main__':
    main()

