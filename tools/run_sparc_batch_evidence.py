#!/usr/bin/env python3
"""
run_sparc_batch_evidence.py - Batch SPARC evidence runs (GR, NFW, ER) and aggregate CSV.

- Reads a list of galaxy IDs (CLI or from MasterSheet in a directory)
- For each galaxy, runs:
  * tools/fit_sparc_gr_evidence.py
  * tools/fit_sparc_nfw_evidence.py
  * tools/fit_sparc_er_evidence.py
- Writes ed_sparc_batch.csv at repo root with columns:
  galaxy, logZ_GR, logZ_NFW, logZ_ER
- Then invokes tools/summarize_ed_sparc_batch.py to render docs.

Usage examples:
  # Run for a specific list
  python tools/run_sparc_batch_evidence.py --sparc-dir external_data/Rotmod_LTG --galaxies NGC3198 NGC2403 NGC2903

  # Or auto-pick first 25 galaxies found in MasterSheet
  python tools/run_sparc_batch_evidence.py --sparc-dir external_data/Rotmod_LTG --limit 25
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import subprocess
import sys
import csv
import os

REPO = Path(__file__).resolve().parents[1]
GR_TOOL = REPO / 'tools' / 'fit_sparc_gr_evidence.py'
NFW_TOOL = REPO / 'tools' / 'fit_sparc_nfw_evidence.py'
ER_TOOL = REPO / 'tools' / 'fit_sparc_er_evidence.py'
CSV_PATH = REPO / 'ed_sparc_batch.csv'
SUMMARY_TOOL = REPO / 'tools' / 'summarize_ed_sparc_batch.py'


def run_cmd(cmd):
    # Force UTF-8 to avoid Windows console encoding issues when tools print Unicode
    env = dict(**os.environ)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, proc.stdout


def parse_json_line_from_output(stdout: str):
    # Try to find a JSON-looking last line; otherwise try whole stdout
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith('{') and ln.endswith('}'):
            try:
                return json.loads(ln)
            except Exception:
                pass
    # Also try to open any printed Saved: path by reading the file
    for ln in reversed(lines):
        if ln.lower().startswith('saved:'):
            path = ln.split(':', 1)[1].strip()
            p = Path(path)
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    pass
    return None


def load_master_galaxies(sparc_dir: Path, limit: int | None):
    # Prefer robust source: filenames *_rotmod.dat
    ids = []
    for p in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = p.name.replace('_rotmod.dat', '')
        if name:
            ids.append(name)
    # Limit
    if limit is not None and limit > 0:
        ids = ids[:limit]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc-dir', required=True, help='Path to SPARC Rotmod_LTG directory with MasterSheet')
    ap.add_argument('--galaxies', nargs='*', default=None, help='Explicit galaxy list (e.g., NGC3198 NGC2403 ...)')
    ap.add_argument('--limit', type=int, default=None, help='If no explicit list, limit to first N from MasterSheet or directory')
    ap.add_argument('--sigma-floor', type=float, default=0.0, help='Velocity error floor (km/s)')
    ap.add_argument('--nlive', type=int, default=800)
    ap.add_argument('--maxcall', type=int, default=150000)
    ap.add_argument('--dlogz-target', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    sparc_dir = Path(args.sparc_dir)
    if not sparc_dir.exists():
        print(f"SPARC dir not found: {sparc_dir}")
        sys.exit(2)

    galaxies = args.galaxies if args.galaxies else load_master_galaxies(sparc_dir, args.limit)
    if not galaxies:
        print('No galaxies found to process.')
        sys.exit(2)

    # Prepare CSV
    CSV_PATH.unlink(missing_ok=True)
    with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['galaxy', 'logZ_GR', 'logZ_NFW', 'logZ_ER'])

    # Loop galaxies
    for gid in galaxies:
        print(f"\n=== {gid} ===")
        row = {'galaxy': gid, 'logZ_GR': None, 'logZ_NFW': None, 'logZ_ER': None}

        # GR
        cmd_gr = [sys.executable, str(GR_TOOL), '--galaxy_id', gid, '--sparc_dir', str(sparc_dir),
                  '--sigma-floor', str(args.sigma_floor), '--nlive', str(args.nlive), '--maxcall', str(args.maxcall),
                  '--dlogz-target', str(args.dlogz_target), '--seed', str(args.seed)]
        rc, out = run_cmd(cmd_gr)
        meta = parse_json_line_from_output(out)
        if not isinstance(meta, dict):
            # Fallback: read expected JSON file path
            p = REPO / 'images' / f"sparc_gr_evidence_{gid.lower()}.json"
            if p.exists():
                try:
                    meta = json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    meta = None
        if isinstance(meta, dict):
            ev = meta.get('evidence', {})
            row['logZ_GR'] = ev.get('logZ')
        else:
            print('Warning: could not parse GR output JSON')

        # NFW
        cmd_nfw = [sys.executable, str(NFW_TOOL), '--galaxy_id', gid, '--sparc_dir', str(sparc_dir),
                   '--sigma-floor', str(args.sigma_floor), '--nlive', str(args.nlive), '--maxcall', str(args.maxcall),
                   '--dlogz-target', str(args.dlogz_target), '--seed', str(args.seed)]
        rc, out = run_cmd(cmd_nfw)
        meta = parse_json_line_from_output(out)
        if not isinstance(meta, dict):
            p = REPO / 'images' / f"sparc_nfw_evidence_{gid.lower()}.json"
            if p.exists():
                try:
                    meta = json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    meta = None
        if isinstance(meta, dict):
            ev = meta.get('evidence', {})
            row['logZ_NFW'] = ev.get('logZ')
        else:
            print('Warning: could not parse NFW output JSON')

        # ER
        cmd_er = [sys.executable, str(ER_TOOL), '--galaxy_id', gid, '--sparc_dir', str(sparc_dir),
                  '--sigma-floor', str(args.sigma_floor), '--nlive', str(args.nlive), '--maxcall', str(args.maxcall),
                  '--dlogz-target', str(args.dlogz_target), '--seed', str(args.seed)]
        rc, out = run_cmd(cmd_er)
        meta = parse_json_line_from_output(out)
        if not isinstance(meta, dict):
            p = REPO / 'images' / f"sparc_er_evidence_{gid.lower()}.json"
            if p.exists():
                try:
                    meta = json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    meta = None
        if isinstance(meta, dict):
            ev = meta.get('evidence', {})
            row['logZ_ER'] = ev.get('logZ')
        else:
            print('Warning: could not parse ER output JSON')

        # Append to CSV
        with CSV_PATH.open('a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([row['galaxy'], row['logZ_GR'], row['logZ_NFW'], row['logZ_ER']])
        print(f"Wrote row: {row}")

    # Generate docs
    rc, out = run_cmd([sys.executable, str(SUMMARY_TOOL)])
    print(out)
    print(f"Batch complete. CSV: {CSV_PATH}")

if __name__ == '__main__':
    main()

