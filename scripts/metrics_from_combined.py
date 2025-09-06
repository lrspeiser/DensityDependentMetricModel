#!/usr/bin/env python3
"""
metrics_from_combined.py

Compute per-run residual metrics (MAE, RMSE, MAPE) from a combined long CSV
produced by scripts/combine_lensing_tables.py.

Usage:
  python -u scripts/metrics_from_combined.py \
    --in results/next_steps/btfr_fix_20250906_lastcross/combined_grid_alpha_only \
    --out results/next_steps/btfr_fix_20250906_lastcross/combined_grid_alpha_only_metrics_by_run.csv

Notes:
- The input can be either the exact long CSV file path or a directory containing
  the long CSV file (named 'combined' without extension, as produced by the combiner).
- Outputs a CSV with columns: run_label,N,MAE_arcsec,RMSE_arcsec,MAPE
- Prints the run_label with the lowest RMSE to stdout as: BEST_ALPHA_TAG=<label>
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

def load_long_csv(path_in: str) -> Tuple[List[Dict[str, str]], List[str]]:
    p = Path(path_in)
    if p.is_dir():
        # the combiner writes the long file as a path (named 'combined') inside the dir
        candidate = p
    else:
        candidate = p
    if candidate.is_dir():
        # choose file named exactly 'combined' in the directory
        fpath = candidate
        # On Windows it may be a file without extension; ensure it exists
        if fpath.is_file():
            pass
        else:
            # try child named 'combined'
            test = candidate / 'combined'
            if not test.exists():
                raise FileNotFoundError(f"Could not find combined long CSV at {candidate}")
            fpath = test
    else:
        fpath = candidate
    rows: List[Dict[str, str]] = []
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for r in reader:
            rows.append(r)
    return rows, headers  # type: ignore[return-value]


def safe_float(s: str) -> float:
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return float('nan')
    except Exception:
        return float('nan')


def compute_metrics(rows: List[Dict[str, str]], run_field: str = 'run_label',
                    obs_field: str = 'theta_E_obs_arcsec', pred_field: str = 'theta_E_RAR_phscaled_arcsec') -> Tuple[List[str], Dict[str, Tuple[int, float, float, float]]]:
    # returns labels, metrics_map[label] = (N, MAE, RMSE, MAPE)
    per: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        lab = r.get(run_field, '')
        if not lab:
            # if combiner used a different field name, try 'label'
            lab = r.get('label', '')
        if not lab:
            continue
        obs = safe_float(r.get(obs_field, 'nan'))
        pred = safe_float(r.get(pred_field, 'nan'))
        if not (math.isfinite(obs) and math.isfinite(pred) and obs > 0):
            continue
        per.setdefault(lab, []).append((pred, obs))
    labels = sorted(per.keys())
    out: Dict[str, Tuple[int, float, float, float]] = {}
    for lab in labels:
        pairs = per[lab]
        if not pairs:
            out[lab] = (0, float('nan'), float('nan'), float('nan'))
            continue
        errs = [abs(p - o) for (p, o) in pairs]
        sq = [e*e for e in errs]
        mape_terms = [abs(p - o) / o for (p, o) in pairs if o != 0]
        N = len(pairs)
        MAE = sum(errs) / N
        RMSE = math.sqrt(sum(sq) / N)
        MAPE = (sum(mape_terms) / len(mape_terms)) if mape_terms else float('nan')
        out[lab] = (N, MAE, RMSE, MAPE)
    return labels, out


def write_metrics_csv(path_out: str, labels: List[str], metrics: Dict[str, Tuple[int, float, float, float]]) -> None:
    p = Path(path_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_label', 'N', 'MAE_arcsec', 'RMSE_arcsec', 'MAPE'])
        for lab in labels:
            N, MAE, RMSE, MAPE = metrics[lab]
            w.writerow([lab, N, f"{MAE:.6f}", f"{RMSE:.6f}", f"{MAPE:.6f}" if math.isfinite(MAPE) else 'nan'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Path to combined long CSV or directory containing it')
    ap.add_argument('--out', required=True, help='Path to write per-run metrics CSV')
    args = ap.parse_args()

    rows, headers = load_long_csv(args.inp)
    labels, metrics = compute_metrics(rows)
    write_metrics_csv(args.out, labels, metrics)

    # Pick the label with the smallest RMSE
    best = None
    best_rmse = float('inf')
    for lab in labels:
        N, MAE, RMSE, MAPE = metrics[lab]
        if math.isfinite(RMSE) and RMSE < best_rmse:
            best_rmse = RMSE
            best = lab
    if best is not None:
        print(f"BEST_ALPHA_TAG={best}")
    else:
        print("BEST_ALPHA_TAG=")

if __name__ == '__main__':
    main()

