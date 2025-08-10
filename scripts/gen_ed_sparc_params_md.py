#!/usr/bin/env python3
"""
Generate a Markdown table of ER hyperparameters per galaxy from ER JSON outputs.
Scans images/sparc_env_fit_*.json and images/sparc_er_evidence_*.json.
Writes docs/ED-SPARC-params.md.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path

FIELDS = [
    'log10_rho_c', 'gamma_exp', 'lambda_max', 'lnT0', 'sigma_lnT', 'w_min'
]


def load_params_from_json(path: Path):
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None, None
    gid = d.get('galaxy_id') or path.stem
    params = d.get('params') or {}
    # Some tools may store under different casing; be forgiving
    rec = {k: params.get(k) for k in FIELDS}
    return gid, rec


def main():
    repo = Path(__file__).resolve().parents[1]
    patterns = [
        repo / 'images' / 'sparc_env_fit_*.json',
        repo / 'images' / 'sparc_er_evidence_*.json',
        repo / 'images' / '*er*evidence_*.json',
    ]
    files = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(str(pat))])

    records = {}
    for p in files:
        gid, rec = load_params_from_json(p)
        if gid is None or rec is None:
            continue
        # Only record if at least one param present
        if any(v is not None for v in rec.values()):
            # Prefer env_fit over other sources if both exist
            prev = records.get(gid)
            if prev is None or p.name.startswith('sparc_env_fit_'):
                records[gid] = rec

    # Build markdown
    out_md = repo / 'docs' / 'ED-SPARC-params.md'
    lines = []
    lines.append('# Extended Data: ER hyperparameters per SPARC galaxy')
    lines.append('')
    headers = ['Galaxy'] + FIELDS
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

    def fmt(x):
        if x is None:
            return ''
        try:
            v = float(x)
            # log10_rho_c, lnT0 can have wider range; keep 3 decimals
            return f'{v:.3f}'
        except Exception:
            return str(x)

    for gid in sorted(records.keys(), key=lambda s: s.lower()):
        rec = records[gid]
        row = [gid] + [fmt(rec.get(k)) for k in FIELDS]
        lines.append('| ' + ' | '.join(row) + ' |')

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote: {out_md} (galaxies={len(records)})')

if __name__ == '__main__':
    main()

