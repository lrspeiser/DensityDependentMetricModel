#!/usr/bin/env python3
"""
Batch SPARC ER/GR/NFW runs with conservative settings and CSV aggregation.
Computes chi2 and ΔlnL proxies (−0.5 Δχ²) for acceptance checks and tables.
"""
from __future__ import annotations
import argparse
import subprocess
import json
import shlex
from pathlib import Path
import sys
import csv

REPO = Path(__file__).resolve().parents[1]


def run_cmd(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO))
    out, err = proc.communicate()
    return proc.returncode, out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace')


def run_one(gal: str, sparc_dir: str, sigma_floor: float) -> dict:
    base = {"galaxy": gal}
    # ER conservative (fit mode for chi2)
    er_cmd = (
        f"python tools/fit_sparc_er_env.py --galaxy_id {gal} --sparc_dir {sparc_dir} "
        f"--mode fit --model er --sigma-floor {sigma_floor} "
        f"--gas-profile RHI --gas-truncation RHI --T-proxy epicyclic --tidal-norm robust "
        f"--use-master-priors --prior-lambda-max 10.0 --prior-wmin-max 0.05 --fit-ml disk bulge"
    )
    code, out, err = run_cmd(er_cmd)
    if code != 0:
        print(f"ER failed for {gal}:\n{err}", file=sys.stderr)
    try:
        er_line = out.strip().splitlines()[-3] if len(out.strip().splitlines())>=3 else out.strip().splitlines()[-1]
        er_obj = json.loads(er_line) if er_line.strip().startswith('{') else None
    except Exception:
        er_obj = None
    # GR
    gr_cmd = (
        f"python tools/fit_sparc_baselines.py --galaxy {gal} --sparc-dir {sparc_dir} --model gr --sigma-floor {sigma_floor}"
    )
    code, out, err = run_cmd(gr_cmd)
    gr_chi2 = None
    if code == 0:
        # Parse chi2 from string
        for line in out.splitlines():
            if line.startswith("GR(baryons-only): chi2="):
                try:
                    chi2_str = line.split("chi2=")[1].split(",")[0]
                    gr_chi2 = float(chi2_str)
                except Exception:
                    pass
    else:
        print(f"GR failed for {gal}:\n{err}", file=sys.stderr)
    # NFW
    nfw_cmd = (
        f"python tools/fit_sparc_baselines.py --galaxy {gal} --sparc-dir {sparc_dir} --model nfw --sigma-floor {sigma_floor}"
    )
    code, out, err = run_cmd(nfw_cmd)
    nfw_chi2 = None
    if code == 0:
        for line in out.splitlines():
            if line.startswith("NFW fit:"):
                try:
                    parts = line.split(',')
                    for p in parts:
                        if p.strip().startswith(' chi2=') or p.strip().startswith('chi2='):
                            nfw_chi2 = float(p.split('=')[1])
                except Exception:
                    pass
    else:
        print(f"NFW failed for {gal}:\n{err}", file=sys.stderr)

    er_chi2 = None
    if er_obj is not None and isinstance(er_obj, dict):
        er_chi2 = float(er_obj.get('chi2', 'nan'))

    row = {
        'galaxy': gal,
        'chi2_GR': gr_chi2,
        'chi2_ER': er_chi2,
        'chi2_NFW': nfw_chi2,
        'dlnL_ER_minus_GR': (None if (er_chi2 is None or gr_chi2 is None) else -0.5*(er_chi2 - gr_chi2)),
        'dlnL_ER_minus_NFW': (None if (er_chi2 is None or nfw_chi2 is None) else -0.5*(er_chi2 - nfw_chi2)),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--sigma-floor', type=float, default=5.0)
    ap.add_argument('--out', default='ed_sparc_batch.csv')
    args = ap.parse_args()

    rows = []
    for g in args.galaxies:
        print(f"[batch] Running {g} ...")
        rows.append(run_one(g, args.sparc_dir, args.sigma_floor))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['galaxy','chi2_GR','chi2_ER','chi2_NFW','dlnL_ER_minus_GR','dlnL_ER_minus_NFW']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved batch CSV: {out_path}")

if __name__ == '__main__':
    main()

