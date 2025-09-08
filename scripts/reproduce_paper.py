#!/usr/bin/env python3
"""
reproduce_paper.py — One-stop script to reproduce the paper's figures and tables
using the RAR-plateau model (xi_rar_plateau) with GR baseline and optional NFW yardstick.

This wraps the next_steps orchestrator with the final flags used in the paper
and collects the key artifacts into a single report.

Usage (example):
  python scripts/reproduce_paper.py \
    --run-dir runs/enhanced_20250805_115400 \
    --sparc-dir external_data/Rotmod_LTG \
    --lensing-csv docs/lensing_targets.csv

Outputs (under results/next_steps/<run> and images/next_steps/<run>):
- Lensing tables/figures (metric-only RAR vs GR; NFW yardstick)
- SPARC overlays and per-galaxy a0 scan
- Solar-System plot/table and PPN table
- Hierarchical a0 (if requested via --hierarchical-a0)
- Milky Way overlay (copied if available under images/rar_plateau_analysis)
- A reproduction report: docs/repro_report.md listing the produced artifacts

Notes:
- This script does not fetch external datasets. Ensure SPARC rotmod files are
  present under external_data/Rotmod_LTG and that docs/lensing_targets.csv is
  populated with measured log10M_star and Re_kpc for desired lenses.
- If a Milky Way overlay figure exists at
  images/rar_plateau_analysis/rar_plateau_mw_comparison_3way.png, it will be
  copied into images/next_steps/<run>/mw_rar_plateau_overlay.png for consistency.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce paper figures/tables (RAR-plateau)")
    ap.add_argument("--run-dir", required=True, help="Run directory used for outputs (e.g., runs/enhanced_20250805_115400)")
    ap.add_argument("--sparc-dir", required=True, help="SPARC rotmod dir (e.g., external_data/Rotmod_LTG)")
    ap.add_argument("--lensing-csv", default="docs/lensing_targets.csv", help="Lens CSV with lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec]")
    ap.add_argument("--sample", default="gold", choices=["gold","q2plus","all"], help="SPARC sample for the orchestrator: gold (default), q2plus (Q<=2), or all")
    ap.add_argument("--preset", default="paper", choices=["paper","pilot","custom"], help="Preset for orchestrator (default: paper)")
    ap.add_argument("--hierarchical-a0", action="store_true", help="Also compute hierarchical a0 MLE over the SPARC subset")
    ap.add_argument("--hierarchical-a0-bayes", action="store_true", help="Run full Bayesian hierarchical posterior for a0 (dynesty)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sparc_dir = Path(args.sparc_dir)
    lens_csv = Path(args.lensing_csv)

    # Orchestrator command (final paper flags)
    cmd = [
        sys.executable,
        "scripts/next_steps_from_run.py",
        "--run-dir", str(run_dir),
        "--sparc-dir", str(sparc_dir),
        "--lensing-sample-csv", str(lens_csv),
        "--preset", str(args.preset),
        "--metric-lensing-only",
        "--density-profile", "sersic",
        "--nfw-enable",
        "--nfw-mass-ratio", "50",
        "--nfw-c", "8",
        "--write-ppn-table",
        "--sample", str(args.sample),
    ]
    if args.hierarchical_a0:
        cmd.append("--hierarchical-a0")
    if args.hierarchical_a0_bayes:
        cmd.append("--hierarchical-a0-bayes")

    run(cmd)

    # Copy Milky Way overlay if available under rar_plateau_analysis
    src_mw = Path("images/rar_plateau_analysis/rar_plateau_mw_comparison_3way.png")
    dst_dir = Path("images") / "next_steps" / run_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_mw = dst_dir / "mw_rar_plateau_overlay.png"
    if src_mw.exists():
        dst_mw.write_bytes(src_mw.read_bytes())
        print(f"Milky Way overlay copied to {dst_mw}")
    else:
        print("Note: MW overlay source not found at", src_mw)

    # Write a short reproduction report
    results_root = Path("results") / "next_steps" / run_dir.name
    report = Path("docs") / "repro_report.md"
    lines = []
    lines.append(f"# Reproduction Report — {run_dir.name}")
    lines.append("")
    lines.append("Milky Way")
    lines.append(f"- Overlay: {dst_mw.as_posix()} (copied if source existed)")
    lines.append("")
    lines.append("SPARC")
    lines.append(f"- a0 summary: { (results_root/'sparc_a0_summary.csv').as_posix() }")
    lines.append(f"- Overlays: images/next_steps/{run_dir.name}/sparc_overlay_<galaxy>.png")
    lines.append("")
    lines.append("Lensing")
    lines.append(f"- Table: { (results_root/'lensing_metric_table.csv').as_posix() }")
    lines.append(f"- θE metrics: { (results_root/'lensing_thetaE_metrics.json').as_posix() }")
    lines.append(f"- Scatter: images/next_steps/{run_dir.name}/lensing_thetaE_pred_vs_obs.png")
    lines.append(f"- ΔΣ stack: { (results_root/'lensing_metric_stack.csv').as_posix() }")
    lines.append(f"- ΔΣ stack plot: images/next_steps/{run_dir.name}/lensing_metric_stack.png")
    lines.append("")
    lines.append("Solar System & PPN")
    lines.append(f"- Table: { (results_root/'solar_system_table.csv').as_posix() }")
    lines.append(f"- Plot: images/next_steps/{run_dir.name}/solar_rar_plateau.png")
    lines.append(f"- PPN: { (results_root/'ppn_table.csv').as_posix() }")
    lines.append("")
    if args.hierarchical_a0:
        lines.append("Hierarchical a0")
        lines.append(f"- Summary JSON: { (results_root/'hierarchical_a0_summary.json').as_posix() }")
        lines.append(f"- Heatmap: images/next_steps/{run_dir.name}/hierarchical_a0_heatmap.png")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

