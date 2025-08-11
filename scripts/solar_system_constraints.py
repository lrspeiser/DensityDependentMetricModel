#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.er_env import xi_env
try:
    from models.er_env import xi_env_exp  # optional exponential screening
    _HAS_EXP = True
except Exception:
    _HAS_EXP = False

# Default Solar-System test points (Msun/kpc^3 units for rho)
DEFAULT_POINTS = [
    {"label": "Earth lab", "rho": 1.0e21, "T": 1.0e3},
    {"label": "Earth orbit", "rho": 1.0e19, "T": 5.0e2},
    {"label": "Jupiter orbit", "rho": 1.0e18, "T": 3.0e2},
    {"label": "Saturn orbit", "rho": 5.0e17, "T": 2.0e2},
    {"label": "Neptune orbit", "rho": 1.0e17, "T": 1.5e2},
    {"label": "Sun photosphere", "rho": 1.0e20, "T": 1.0e3},
]

# Constraint bands (rough canonical values)
CASSINI_MAX = 2.3e-5  # |gamma-1| limit; use as a reference horizontal line for |xi-1|
LLR_MAX = 1.0e-10     # placeholder tight constraint scale


def load_params_from_er_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    p = meta.get('params', {})
    # Map to expected keys
    return {
        'log10_rho_c': float(p.get('log10_rho_c', 15.0)),
        'gamma_exp': float(p.get('gamma_exp', 3.0)),
        'lambda_max': float(p.get('lambda_max', 4.0)),
        'lnT0': float(p.get('lnT0', 0.0)),
        'sigma_lnT': float(p.get('sigma_lnT', 0.8)),
        'w_min': float(p.get('w_min', 0.02)),
        'rho_screen': p.get('rho_screen', 'power'),
        'galaxy_id': meta.get('galaxy_id', 'Unknown'),
    }


def compute_xi_minus_1(params: Dict[str, Any], points: List[Dict[str, Any]], use_exp: bool = False) -> List[Dict[str, Any]]:
    rho_c = 10.0 ** params['log10_rho_c']
    gamma_exp = params['gamma_exp']
    lambda_max = params['lambda_max']
    T0 = float(np.exp(params['lnT0']))
    sigma_lnT = params['sigma_lnT']
    w_min = params['w_min']
    rows = []
    for pt in points:
        rho = float(pt['rho'])
        T = float(pt.get('T', T0))
        xi = xi_env_exp(rho, T, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min) if (use_exp and _HAS_EXP) else xi_env(rho, T, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min)
        rows.append({
            'label': pt['label'],
            'rho': rho,
            'T': T,
            'xi': float(xi),
            'abs_dev': float(abs(xi - 1.0)),
        })
    return rows


def plot_constraints(rows_power: List[Dict[str, Any]], rows_exp: List[Dict[str, Any]] | None, out_png: Path, title: str) -> None:
    labels = [r['label'] for r in rows_power]
    x = np.arange(len(labels))
    y_p = [r['abs_dev'] for r in rows_power]
    plt.figure(figsize=(10, 5))
    plt.semilogy(x, y_p, 'ro-', label='|xi-1| (power S_rho)')
    if rows_exp is not None:
        y_e = [r['abs_dev'] for r in rows_exp]
        plt.semilogy(x, y_e, 'bs--', label='|xi-1| (exp S_rho)')
    plt.axhline(CASSINI_MAX, color='k', ls=':', label='Cassini γ limit')
    plt.axhline(LLR_MAX, color='gray', ls='--', alpha=0.6, label='LLR scale (indicative)')
    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel('|xi - 1|')
    plt.title(title)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'Saved: {out_png}')


def write_markdown_table(rows_power: List[Dict[str, Any]], rows_exp: List[Dict[str, Any]] | None, out_md: Path, header: str, params_src: str) -> None:
    lines = []
    lines.append(header)
    lines.append("")
    lines.append(f"Parameters source: {params_src}")
    lines.append("")
    cols = ["Environment", "rho [Msun/kpc^3]", "T", "|xi-1| power"]
    if rows_exp is not None:
        cols.append("|xi-1| exp")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for i, r in enumerate(rows_power):
        row = [r['label'], f"{r['rho']:.3e}", f"{r['T']:.2e}", f"{r['abs_dev']:.3e}"]
        if rows_exp is not None:
            row.append(f"{rows_exp[i]['abs_dev']:.3e}")
        lines.append("| " + " | ".join(row) + " |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding='utf-8')
    print(f'Saved: {out_md}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--er-json', required=True, help='Path to a representative ER fit JSON (params source)')
    ap.add_argument('--points-json', default=None, help='Optional JSON with a list of {label,rho,T}')
    ap.add_argument('--use-exp', action='store_true', help='Compute exponential screening variant in addition to power-law')
    ap.add_argument('--out-png', default='images/solar_system_constraints.png')
    ap.add_argument('--out-md', default='docs/table_solar_screening.md')
    args = ap.parse_args()

    params = load_params_from_er_json(Path(args.er_json))

    # Load points
    pts = DEFAULT_POINTS
    if args.points_json:
        try:
            pts = json.loads(Path(args.points_json).read_text(encoding='utf-8'))
        except Exception:
            print('Warning: failed to read points JSON; using defaults')

    rows_power = compute_xi_minus_1(params, pts, use_exp=False)
    rows_exp = compute_xi_minus_1(params, pts, use_exp=True) if args.use_exp and _HAS_EXP else None

    title = f"Solar-System constraints with galaxy-fit params ({params.get('galaxy_id','')})"
    plot_constraints(rows_power, rows_exp, Path(args.out_png), title)
    header = "# Solar-System screening with galaxy-fit parameters"
    write_markdown_table(rows_power, rows_exp, Path(args.out_md), header, Path(args.er_json).name)


if __name__ == '__main__':
    main()
