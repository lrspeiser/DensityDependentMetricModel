#!/usr/bin/env python3
# Appends a metrics summary to docs/next_steps.md

import csv
from pathlib import Path
from datetime import datetime

BASE = Path('results/next_steps/btfr_fix_20250906_lastcross/combined_metrics')
DOC = Path('docs/next_steps.md')
IMG_BASE = Path('images/next_steps/btfr_fix_20250906_lastcross/metrics')


def best_row(path: Path, key: str = 'rms_rel'):
    if not path.exists():
        return None
    with path.open() as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        return None
    rows_num = []
    for r in rows:
        try:
            rr = float(r.get(key, 'nan'))
        except Exception:
            continue
        rows_num.append((rr, r))
    if not rows_num:
        return None
    rows_num.sort(key=lambda t: t[0])
    return rows_num[0][1]


def fmt(val):
    try:
        return f"{float(val):.4g}"
    except Exception:
        return str(val)


def main():
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    lines = []
    lines.append('\n')
    lines.append('## Combined metrics summary')
    lines.append(f"Generated: {ts}")
    lines.append('')

    for profile in ['hernquist', 'jaffe']:
        alpha_csv = BASE / f'metrics_alpha_only_{profile}.csv'
        zeta_csv = BASE / f'metrics_zeta_alpha2_{profile}.csv'
        br_a = best_row(alpha_csv)
        br_z = best_row(zeta_csv)
        lines.append(f"### {profile.capitalize()}")
        if br_a:
            lines.append(f"- Alpha-only (zeta=0, env=constant): best alpha={fmt(br_a['alpha'])}, RMS_rel={fmt(br_a['rms_rel'])}, N={int(float(br_a['n']))}")
        else:
            lines.append("- Alpha-only: no data")
        if br_z:
            lines.append(f"- Tapered zeta @ alpha=2.0: best zeta={fmt(br_z['zeta'])}, RMS_rel={fmt(br_z['rms_rel'])}, N={int(float(br_z['n']))}")
        else:
            lines.append("- Tapered zeta @ alpha=2.0: no data")
        lines.append('')

    lines.append('Artifacts:')
    lines.append(f"- All runs: {BASE / 'metrics_all_runs.csv'}")
    lines.append(f"- Plots: {IMG_BASE}")
    lines.append(f"  - RMS vs alpha (per profile): rms_rel_vs_alpha_*.png")
    lines.append(f"  - RMS vs zeta at alpha=2.0 (per profile): rms_rel_vs_zeta_alpha2_*.png")
    lines.append('')
    lines.append('Notes:')
    lines.append('- Lensing environment scaling uses clamp to keep (1 + zeta_env_lens * f(R)) >= 0, avoiding unphysical negative convergence for negative zeta.')
    lines.append('- Einstein solver sometimes reported non-monotone <Sigma>(R); monotone envelope was applied during integration (expected for noisy grids).')
    lines.append('')

    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] Appended metrics summary to {DOC}")


if __name__ == '__main__':
    main()

