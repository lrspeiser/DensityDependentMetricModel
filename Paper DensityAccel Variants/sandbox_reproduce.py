#!/usr/bin/env python3
# Orchestrate sandbox runs for density- and acceleration-gated variants across
# clusters, SPARC, lensing, MW Kz, and ephemeris. Produces a unified report.
#
# This script only uses the helper scripts under "Paper DensityAccel Variants/"
# and does not modify production code or figures.

from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = REPO_ROOT / 'Paper DensityAccel Variants'
RESULTS_DIR = SANDBOX_DIR / 'results'
IMAGES_DIR = SANDBOX_DIR / 'images'


class StepError(Exception):
    pass


def run(cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print('>>', ' '.join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and cp.returncode != 0:
        raise StepError(f"Command failed: {' '.join(cmd)}")
    return cp


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def step_clusters(args) -> Dict[str, Any]:
    out = {}
    # Hybrid/density variant using ACCEPT and optional stars
    cl_out = RESULTS_DIR / 'cluster_hybrid'
    cl_img = IMAGES_DIR / 'cluster_hybrid'
    cl_out.mkdir(parents=True, exist_ok=True)
    cl_img.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SANDBOX_DIR / 'cluster_dg_ag_variants.py'),
        '--accept', args.accept,
        '--results', str(cl_out),
        '--images', str(cl_img),
        '--gate', args.cluster_gate,
        '--a0', str(args.a0),
        '--rho-c', str(args.rho_c),
        '--rho-gamma', str(args.gamma),
        '--zeta', str(args.zeta),
        '--Dmax', str(args.Dmax),
    ]
    if args.stars_csv and Path(args.stars_csv).exists():
        cmd += ['--stars-csv', args.stars_csv]
    run(cmd)
    out['hybrid_results'] = str(cl_out)
    out['hybrid_images'] = str(cl_img)

    if args.cluster_compare:
        # Compute GR vs optional gate metrics (median per-cluster RMS) vs CLASH NFW
        compare_out = RESULTS_DIR / 'cluster_compare'
        compare_out.mkdir(parents=True, exist_ok=True)
        out_json = compare_out / 'compare_metrics.json'
        cmd2 = [
            sys.executable, str(SANDBOX_DIR / 'cluster_compare_metrics.py'),
            '--accept', args.accept,
            '--out-json', str(out_json),
            '--mu-e', str(args.mu_e),
            '--xmin', str(args.xmin), '--xmax', str(args.xmax),
            '--equal-cluster-weight',
        ]
        if args.cluster_gate in ('density', 'hybrid'):
            cmd2 += [
                '--gate', args.cluster_gate if args.cluster_gate in ('density','hybrid') else 'density',
                '--a0', str(args.a0),
                '--rho-c', str(args.rho_c),
                '--gamma', str(args.gamma),
                '--zeta', str(args.zeta),
                '--xi-max', str(args.Dmax),
            ]
        run(cmd2)
        out['compare_metrics_json'] = str(out_json)
    return out


def step_sparc(args) -> Dict[str, Any]:
    out = {}
    # Convert rotmods to CSV (if requested)
    csv_dir = SANDBOX_DIR / 'sparc_csv'
    if args.convert_sparc:
        csv_dir.mkdir(parents=True, exist_ok=True)
        cmd_conv = [
            sys.executable, str(SANDBOX_DIR / 'sparc_rotmod_to_csv.py'),
            '--src-dir', args.sparc_rotmods,
            '--out-dir', str(csv_dir),
        ]
        if args.sparc_limit > 0:
            cmd_conv += ['--limit', str(args.sparc_limit)]
        run(cmd_conv)
    else:
        csv_dir = Path(args.sparc_csv or (SANDBOX_DIR / 'sparc_csv'))

    # Build rho proxy if density/hybrid requested
    if args.run_sparc_density:
        rho_csv = SANDBOX_DIR / 'sparc_rho_proxy.csv'
        if args.build_rho_proxy:
            cmd_rho = [
                sys.executable, str(SANDBOX_DIR / 'sparc_rho_proxy_from_dat.py'),
                '--src-dir', args.sparc_rotmods,
                '--out-csv', str(rho_csv),
            ]
            if args.sparc_limit > 0:
                cmd_rho += ['--limit', str(args.sparc_limit)]
            run(cmd_rho)
        else:
            rho_csv = Path(args.rho_csv or rho_csv)
        out['sparc_rho_csv'] = str(rho_csv)
    else:
        rho_csv = None

    # Run accel gate
    if args.run_sparc_accel:
        outdir_accel = RESULTS_DIR / 'sparc_accel'
        outdir_accel.mkdir(parents=True, exist_ok=True)
        cmd_accel = [
            sys.executable, str(SANDBOX_DIR / 'sparc_variants.py'),
            '--sparc-rotmods', str(csv_dir),
            '--outdir', str(outdir_accel),
            '--gate', 'accel',
        ] + (['--galaxies'] + args.galaxies if args.galaxies else [])
        run(cmd_accel)
        out['sparc_accel_dir'] = str(outdir_accel)

    # Run density gate
    if args.run_sparc_density:
        outdir_den = RESULTS_DIR / 'sparc_density'
        outdir_den.mkdir(parents=True, exist_ok=True)
        cmd_den = [
            sys.executable, str(SANDBOX_DIR / 'sparc_variants.py'),
            '--sparc-rotmods', str(csv_dir),
            '--outdir', str(outdir_den),
            '--gate', 'density',
            '--rho-csv', str(rho_csv),
            '--rho-c', str(args.rho_c),
            '--gamma', str(args.gamma),
            '--Dmax', str(args.Dmax),
        ] + (['--galaxies'] + args.galaxies if args.galaxies else [])
        run(cmd_den)
        out['sparc_density_dir'] = str(outdir_den)
    return out


def step_lensing(args) -> Dict[str, Any]:
    out = {}
    out_json = RESULTS_DIR / 'lensing_density.json'
    cmd = [
        sys.executable, str(SANDBOX_DIR / 'lensing_variants.py'),
        '--gate', 'density',
        '--rho-c', str(args.rho_c), '--gamma', str(args.gamma), '--Dmax', str(args.Dmax),
        '--log10Mstar', str(args.lens_log10M), '--Re_kpc', str(args.lens_Re_kpc),
        '--Sigma_crit_cgs', str(args.lens_sigma_cr),
        '--out', str(out_json),
    ]
    run(cmd)
    out['lensing_json'] = str(out_json)
    return out


def step_ephemeris(args) -> Dict[str, Any]:
    out = {}
    cmd = [
        sys.executable, str(SANDBOX_DIR / 'ephemeris_variants.py'),
        '--gate', 'density', '--rho-c', str(args.rho_c), '--gamma', str(args.gamma), '--Dmax', str(args.Dmax),
        '--rho-env', str(args.ephem_rho_env),
    ]
    run(cmd)
    out['ephemeris'] = {'rho_env': args.ephem_rho_env}
    return out


def step_mw_kz(args) -> Dict[str, Any]:
    out = {}
    outdir = RESULTS_DIR / 'mw_kz'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SANDBOX_DIR / 'mw_kz_variants.py'),
        '--gate', 'density', '--rho-c', str(args.rho_c), '--gamma', str(args.gamma), '--Dmax', str(args.Dmax),
        '--outdir', str(outdir),
    ]
    run(cmd)
    out['mw_kz_dir'] = str(outdir)
    return out


def write_report(artifacts: Dict[str, Any]) -> None:
    # Minimal unified report that references produced artifacts
    md = [
        '# Sandbox Report — Density vs Acceleration Gating',
        '',
        'This report was generated by sandbox_reproduce.py and summarizes the artifacts written under "Paper DensityAccel Variants/results".',
        '',
    ]
    if 'clusters' in artifacts:
        cl = artifacts['clusters']
        md += [
            '## Clusters (hybrid/density)',
            f"- Results dir: {cl.get('hybrid_results','')}\n- Images dir: {cl.get('hybrid_images','')}",
        ]
        if 'compare_metrics_json' in cl:
            md.append(f"- Compare metrics JSON (GR vs gate): {cl['compare_metrics_json']}")
            try:
                data = json.loads(Path(cl['compare_metrics_json']).read_text())
                glob = data.get('global', {})
                conf = data.get('config', {})
                md.append(f"  - Median RMS (GR): {glob.get('median_rms_gr_dex','nan')}")
                if 'median_rms_gate_dex' in glob:
                    md.append(f"  - Median RMS (gate): {glob.get('median_rms_gate_dex','nan')} with gate={conf.get('gate')} params={conf.get('params')}")
            except Exception:
                pass
        md.append('')

    if 'sparc' in artifacts:
        sp = artifacts['sparc']
        md += ['## SPARC',]
        if 'sparc_accel_dir' in sp:
            md.append(f"- Accel-gate models dir: {sp['sparc_accel_dir']}")
        if 'sparc_density_dir' in sp:
            md.append(f"- Density-gate models dir: {sp['sparc_density_dir']}")
        if 'sparc_rho_csv' in sp:
            md.append(f"- Rho proxy CSV: {sp['sparc_rho_csv']}")
        md.append('')

    if 'lensing' in artifacts:
        md += ['## Lensing', f"- JSON: {artifacts['lensing'].get('lensing_json','')}", '']
    if 'ephemeris' in artifacts:
        md += ['## Ephemeris', f"- rho_env: {artifacts['ephemeris'].get('rho_env','')}", '']
    if 'mw_kz' in artifacts:
        md += ['## Milky Way Kz', f"- Dir: {artifacts['mw_kz'].get('mw_kz_dir','')}", '']

    (SANDBOX_DIR / 'sandbox_report.md').write_text('\n'.join(md), encoding='utf-8')
    (SANDBOX_DIR / 'sandbox_report.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Sandbox orchestrator for density vs acceleration gate variants')
    # Data roots
    ap.add_argument('--accept', default=str(REPO_ROOT / 'external_data' / 'accept_database.dat'))
    ap.add_argument('--stars-csv', default=str(REPO_ROOT / 'external_data' / 'clash_stars.csv'))
    ap.add_argument('--sparc-rotmods', default=str(REPO_ROOT / 'external_data' / 'Rotmod_LTG'))
    ap.add_argument('--sparc-csv', default='')
    ap.add_argument('--galaxies', nargs='*', default=['CamB', 'D631-7'])
    ap.add_argument('--sparc-limit', type=int, default=20)

    # Gates and physics params
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--gamma', type=float, default=1.5)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)

    # Cluster compare config
    ap.add_argument('--mu-e', type=float, default=1.17)
    ap.add_argument('--xmin', type=float, default=0.05)
    ap.add_argument('--xmax', type=float, default=0.8)
    ap.add_argument('--cluster-gate', default='hybrid', choices=['density','hybrid','accel'])

    # Lensing test params
    ap.add_argument('--lens-log10M', type=float, default=11.6)
    ap.add_argument('--lens-Re-kpc', type=float, default=8.0)
    ap.add_argument('--lens-sigma-cr', type=float, default=1.5e9)

    # Ephemeris env
    ap.add_argument('--ephem-rho-env', type=float, default=1e-21)

    # Toggles
    ap.add_argument('--run-clusters', action='store_true')
    ap.add_argument('--cluster-compare', action='store_true')
    ap.add_argument('--run-sparc-accel', action='store_true')
    ap.add_argument('--run-sparc-density', action='store_true')
    ap.add_argument('--convert-sparc', action='store_true')
    ap.add_argument('--build-rho-proxy', action='store_true')
    ap.add_argument('--run-lensing', action='store_true')
    ap.add_argument('--run-ephemeris', action='store_true')
    ap.add_argument('--run-mw-kz', action='store_true')
    ap.add_argument('--report-only', action='store_true')

    args = ap.parse_args()

    ensure_dirs()

    artifacts: Dict[str, Any] = {}

    if args.run_clusters:
        artifacts['clusters'] = step_clusters(args)
    if args.run_sparc_accel or args.run_sparc_density:
        artifacts['sparc'] = step_sparc(args)
    if args.run_lensing:
        artifacts['lensing'] = step_lensing(args)
    if args.run_ephemeris:
        artifacts['ephemeris'] = step_ephemeris(args)
    if args.run_mw_kz:
        artifacts['mw_kz'] = step_mw_kz(args)

    # Always (re)write report unless explicitly disabled
    if not (args.report_only and not artifacts):
        write_report(artifacts)
        print('Wrote', SANDBOX_DIR / 'sandbox_report.md')
        print('Wrote', SANDBOX_DIR / 'sandbox_report.json')
    else:
        print('Report-only mode with no new artifacts; skipping report write.')


if __name__ == '__main__':
    main()
