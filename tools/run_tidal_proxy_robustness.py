#!/usr/bin/env python3
"""
Run ER fit-mode robustness checks across tidal proxy choices for selected galaxies.
- For each galaxy, runs tools/fit_sparc_er_env.py in fit mode with:
  - T-proxy in {curvature, shear, epicyclic}
  - robust norm, RHI gas truncation
  - mild sigma-floor and M/L priors matching paper setup
- Outputs docs/robustness_tidal_proxy.md with a small table per galaxy.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import sys
import json

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable or 'python'


def run_one(gal: str, proxy: str, sigma_floor: float = 5.0) -> dict:
    cmd = [
        PY, str(REPO / 'tools' / 'fit_sparc_er_env.py'),
        '--galaxy_id', gal,
        '--sparc_dir', 'external_data/Rotmod_LTG',
        '--mode', 'fit', '--model', 'er', '--sigma-floor', str(sigma_floor),
        '--gas-profile', 'RHI', '--gas-truncation', 'RHI',
        '--T-proxy', proxy, '--tidal-norm', 'robust',
        '--use-master-priors', '--prior-lambda-max', '10.0', '--prior-wmin-max', '0.05',
        '--fit-ml', 'disk', 'bulge'
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout.strip().splitlines()
    # Try to find a JSON line in last few lines
    result = {'ok': proc.returncode == 0, 'chi2': None, 'error': None}
    if proc.returncode != 0:
        result['error'] = out[-1] if out else 'failed'
        return result
    # Fallback: parse chi2 from final print lines if JSON not available
    for line in reversed(out[-10:]):
        if 'chi2' in line.lower():
            # crude extraction of last float on line
            import re
            m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if m:
                try:
                    result['chi2'] = float(m[-1])
                except Exception:
                    pass
            break
    return result


def main():
    galaxies = ['NGC3198', 'NGC2403']
    proxies = ['curvature', 'shear', 'epicyclic']
    rows = []
    for g in galaxies:
        for p in proxies:
            print(f"[robustness] {g} / {p} ...")
            res = run_one(g, p)
            rows.append({'galaxy': g, 'proxy': p, **res})
    # Write markdown
    out_md = REPO / 'docs' / 'robustness_tidal_proxy.md'
    lines = ["# Tidal Proxy Robustness (Fit Mode)",""]
    for g in galaxies:
        lines.append(f"## {g}")
        lines.append("| Proxy | chi2 | Status |")
        lines.append("|---|---:|---|")
        for p in proxies:
            r = next(x for x in rows if x['galaxy']==g and x['proxy']==p)
            status = 'OK' if r['ok'] else f"FAIL ({r.get('error','')})"
            chi2s = '—' if r['chi2'] is None else f"{r['chi2']:.2f}"
            lines.append(f"| {p} | {chi2s} | {status} |")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote {out_md}")


if __name__ == '__main__':
    main()
