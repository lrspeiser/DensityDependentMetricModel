#!/usr/bin/env python3
"""
regenerate_pdf.py - Convert Markdown tables to PDF for paper_release

Usage:
  python paper_release/scripts/regenerate_pdf.py \
    --md paper_release/tables/ablation_summary.md \
    --out paper_release/tables/ablation_summary.pdf

Requires pandoc installed and available on PATH.
"""
import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default='paper_release/tables/ablation_summary.md')
    ap.add_argument('--out', default='paper_release/tables/ablation_summary.pdf')
    args = ap.parse_args()

    md = Path(args.md)
    pdf = Path(args.out)
    pdf.parent.mkdir(parents=True, exist_ok=True)

    cmd = ['pandoc', str(md), '-o', str(pdf), '--from=markdown', '--pdf-engine=xelatex']
    run(cmd)
    print(f"Wrote {pdf}")


if __name__ == '__main__':
    main()
