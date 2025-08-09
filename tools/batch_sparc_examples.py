#!/usr/bin/env python3
"""
Batch runner for SPARC examples: fits ER parameters for selected galaxies and saves plots.
CPU-only; quick. Requires the Rotmod_LTG files exist locally.

Usage:
  python tools/batch_sparc_examples.py

It will attempt NGC 3198 and M33 (NGC 598) if their files exist.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

def run(cmd):
    print('> ' + ' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"Command failed with code {proc.returncode}")
    return proc.returncode

REPO = Path(__file__).resolve().parents[1]
rot = REPO / 'external_data' / 'Rotmod_LTG'

cases = [
    (rot / 'NGC3198_rotmod.dat', 'NGC 3198', REPO / 'images' / 'sparc_ngc3198_fit.png'),
    (rot / 'NGC0598_rotmod.dat', 'M33', REPO / 'images' / 'sparc_m33_fit.png'),
]

def main():
    for path, name, out in cases:
        if path.exists():
            run([sys.executable, str(REPO/'tools'/'fit_sparc_er.py'),
                 '--file', str(path), '--name', name, '--out', str(out)])
        else:
            print(f"Skipping {name}: file not found: {path}")

if __name__ == '__main__':
    main()
