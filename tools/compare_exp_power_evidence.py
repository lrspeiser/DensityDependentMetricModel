#!/usr/bin/env python3
"""
Compare evidence between exp_hi and power_hi env-ER variants for selected galaxies.
Reads JSON sidecars from images/ and writes a markdown summary to docs/solar.
Usage:
  python tools/compare_exp_power_evidence.py NGC3198 NGC2403 NGC2841 NGC6946 NGC5055
"""
from __future__ import annotations
import sys
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
IM = REPO / 'images'
OUT_MD = REPO / 'docs' / 'solar' / 'solar_evidence_deltas.md'

def load_logZ(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            j = json.load(f)
        ev = j.get('evidence', {})
        return ev.get('logZ', None), ev.get('logZ_err', None)
    except Exception:
        return None, None


def main(argv: list[str]):
    gals = argv or ['NGC3198','NGC2403','NGC2841','NGC6946','NGC5055']
    rows = []
    for g in gals:
        p_exp = IM / f'sparc_env_fit_{g}_exp_hi.json'
        p_pow = IM / f'sparc_env_fit_{g}_power_hi.json'
        z_exp, e_exp = load_logZ(p_exp)
        z_pow, e_pow = load_logZ(p_pow)
        d = None if (z_exp is None or z_pow is None) else (z_pow - z_exp)
        rows.append((g, z_exp, e_exp, z_pow, e_pow, d))
    # Write markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# ΔlogZ between power_hi and exp_hi (env-ER)')
    lines.append('')
    lines.append('| Galaxy | logZ(exp_hi) | logZ_err | logZ(power_hi) | logZ_err | ΔlogZ(power−exp) |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    for g, ze, ee, zp, ep, d in rows:
        lines.append(f'| {g} | {fmt(ze)} | {fmt(ee)} | {fmt(zp)} | {fmt(ep)} | {fmt(d)} |')
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {OUT_MD}')

if __name__ == '__main__':
    main(sys.argv[1:])

