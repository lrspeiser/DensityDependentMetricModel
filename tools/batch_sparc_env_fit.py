#!/usr/bin/env python3
"""
Batch SPARC ER/GR/NFW runs with conservative settings and CSV aggregation.
- Fit mode: computes chi2 and ΔlnL proxies (−0.5 Δχ²) for acceptance checks.
- Evidence mode: runs dynesty evidence tools for GR, NFW, and ER, aggregating logZs and Bayes factors.
"""
from __future__ import annotations
import argparse
import subprocess
import json
from pathlib import Path
import sys
import csv

REPO = Path(__file__).resolve().parents[1]


def run_cmd(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO))
    out, err = proc.communicate()
    return proc.returncode, out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace')


def run_one_fit(gal: str, sparc_dir: str, sigma_floor: float) -> dict:
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


def run_one_evidence(gal: str, sparc_dir: str, sigma_floor: float, nlive: int, maxcall: int, dlogz: float, seed: int) -> dict:
    gl = gal.lower()
    # GR evidence
    gr_cmd = (
        f"python tools/fit_sparc_gr_evidence.py --galaxy_id {gal} --sparc_dir {sparc_dir} "
        f"--sigma-floor {sigma_floor} --use-master-priors --mode evidence "
        f"--nlive {nlive} --maxcall {maxcall} --dlogz-target {dlogz} --seed {seed}"
    )
    code, out, err = run_cmd(gr_cmd)
    if code != 0:
        print(f"GR evidence failed for {gal}:\n{err}", file=sys.stderr)
    # NFW evidence
    nfw_cmd = (
        f"python tools/fit_sparc_nfw_evidence.py --galaxy_id {gal} --sparc_dir {sparc_dir} "
        f"--sigma-floor {sigma_floor} --use-master-priors --mode evidence "
        f"--nlive {nlive} --maxcall {maxcall} --dlogz-target {dlogz} --seed {seed}"
    )
    code, out, err = run_cmd(nfw_cmd)
    if code != 0:
        print(f"NFW evidence failed for {gal}:\n{err}", file=sys.stderr)
    # ER evidence
    er_cmd = (
        f"python tools/fit_sparc_er_env.py --galaxy_id {gal} --sparc_dir {sparc_dir} "
        f"--mode evidence --model er --sigma-floor {sigma_floor} "
        f"--gas-profile RHI --gas-truncation RHI --T-proxy epicyclic --tidal-norm robust "
        f"--use-master-priors --prior-lambda-max 10.0 --prior-wmin-max 0.05 --fit-ml disk bulge "
        f"--nlive {nlive} --maxcall {maxcall} --dlogz-target {dlogz} --seed {seed}"
    )
    code, out, err = run_cmd(er_cmd)
    if code != 0:
        print(f"ER evidence failed for {gal}:\n{err}", file=sys.stderr)

    # Read JSON sidecars
    gr_json = REPO / 'images' / f'sparc_gr_evidence_{gl}.json'
    nfw_json = REPO / 'images' / f'sparc_nfw_evidence_{gl}.json'
    er_json = REPO / 'images' / f'sparc_env_fit_{gl}.json'
    def read_json(p: Path):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    Jgr = read_json(gr_json)
    Jnfw = read_json(nfw_json)
    Jer = read_json(er_json)

    def get_ev(J, key='evidence'):
        if not J or key not in J or J[key] is None:
            return None, None
        return J[key].get('logZ', None), J[key].get('logZ_err', None)

    z_gr, dz_gr = get_ev(Jgr)
    z_nfw, dz_nfw = get_ev(Jnfw)
    z_er, dz_er = get_ev(Jer)

    row = {
        'galaxy': gal,
        'logZ_GR': z_gr,
        'logZ_GR_err': dz_gr,
        'logZ_NFW': z_nfw,
        'logZ_NFW_err': dz_nfw,
        'logZ_ER': z_er,
        'logZ_ER_err': dz_er,
        'dlogZ_ER_minus_GR': (None if (z_er is None or z_gr is None) else (z_er - z_gr)),
        'dlogZ_ER_minus_NFW': (None if (z_er is None or z_nfw is None) else (z_er - z_nfw)),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--sigma-floor', type=float, default=5.0)
    ap.add_argument('--out', default='ed_sparc_batch.csv')
    ap.add_argument('--mode', choices=['fit','evidence'], default='fit')
    # Evidence knobs
    ap.add_argument('--nlive', type=int, default=1000)
    ap.add_argument('--maxcall', type=int, default=200000)
    ap.add_argument('--dlogz-target', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = []
    for g in args.galaxies:
        print(f"[batch] Running {g} ...")
        if args.mode == 'fit':
            rows.append(run_one_fit(g, args.sparc_dir, args.sigma_floor))
        else:
            rows.append(run_one_evidence(g, args.sparc_dir, args.sigma_floor, args.nlive, args.maxcall, args.dlogz_target, args.seed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'fit':
        cols = ['galaxy','chi2_GR','chi2_ER','chi2_NFW','dlnL_ER_minus_GR','dlnL_ER_minus_NFW']
    else:
        cols = ['galaxy','logZ_GR','logZ_GR_err','logZ_NFW','logZ_NFW_err','logZ_ER','logZ_ER_err','dlogZ_ER_minus_GR','dlogZ_ER_minus_NFW']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved batch CSV: {out_path}")

if __name__ == '__main__':
    main()
