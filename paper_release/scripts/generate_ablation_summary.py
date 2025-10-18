#!/usr/bin/env python3
"""
generate_ablation_summary.py - Generate combined ablation/sensitivity summary tables

Reads the outputs from run_full_tuning_pipeline.py and produces human-readable
Markdown tables for the paper_release results section.

Usage:
    python paper_release/scripts/generate_ablation_summary.py \
      --input results/combined_analysis/pipeline_summary.json \
      --out paper_release/tables/ablation_summary.md
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def load_summary(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_md_table(df: pd.DataFrame, columns: list[str], title: str) -> str:
    header = f"\n## {title}\n\n"
    table = df[columns].to_markdown(index=False)
    return header + table + "\n"


def generate_markdown(summary_json: Path, out_md: Path, csv_details: Path | None = None) -> None:
    summary = load_summary(summary_json)
    details_csv = summary_json.parent / 'detailed_results.csv'
    if csv_details and Path(csv_details).exists():
        details_csv = Path(csv_details)

    df = pd.read_csv(details_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("# Ablation and Sensitivity Summary\n")
    parts.append(f"Generated from {summary_json} and {details_csv}\n\n")

    # High-level performance
    perf = summary.get('performance_summary', {})
    parts.append("## Overview\n\n")
    parts.append(f"- RAR scatter (range): {perf.get('rar_scatter_range',[None,None])}\n")
    parts.append(f"- RAR scatter (mean): {perf.get('rar_scatter_mean', None)}\n")
    parts.append(f"- Cluster error (range): {perf.get('cluster_error_range',[None,None])}\n")
    parts.append(f"- Cluster error (mean): {perf.get('cluster_error_mean', None)}\n")
    parts.append(f"- Solar System pass rate: {perf.get('solar_system_pass_rate', None):.2f}\n\n")

    # Gate ablation subset table
    ablations = df[df['run_id'].str.startswith('no_') | (df['run_id']=='baseline')].copy()
    if not ablations.empty:
        ablations = ablations[['run_id','gate_config','rar_scatter','cluster_error','solar_system_pass']]
        parts.append(make_md_table(ablations,
                                   ['run_id','gate_config','rar_scatter','cluster_error','solar_system_pass'],
                                   'Gate Ablation Results'))

    # Sensitivity entries (heuristic: runs with pattern name_±% in run_id)
    sens = df[df['run_id'].str.contains('_\+|_\-', regex=True)].copy()
    if not sens.empty:
        parts.append(make_md_table(sens,
                                   ['run_id','rar_scatter','cluster_error','L_0','beta_bulge','alpha_shear','gamma_bar','n_coh','p'],
                                   'Sensitivity Results'))

    # Best performers
    best_rar = min(df['rar_scatter']) if 'rar_scatter' in df else None
    if best_rar is not None:
        best_rows = df[df['rar_scatter'] == best_rar]
        parts.append("\n## Best RAR Model(s)\n\n")
        parts.append(best_rows.to_markdown(index=False))

    out_md.write_text("\n".join(parts), encoding='utf-8')
    print(f"Wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=False, default='results/combined_analysis/pipeline_summary.json')
    ap.add_argument('--out', required=False, default='paper_release/tables/ablation_summary.md')
    ap.add_argument('--details_csv', required=False, default=None)
    args = ap.parse_args()

    generate_markdown(Path(args.input), Path(args.out), Path(args.details_csv) if args.details_csv else None)


if __name__ == '__main__':
    main()
