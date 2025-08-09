#!/usr/bin/env python3
"""
Analyze Dynesty run summaries for DDMM runs and load results into DuckDB.

- Reads run_summary_enhanced.json from a run directory
- Emits a compact summary JSON under results/<run_name>/summary_min.json
- Optionally compares multiple runs (e.g., tidal_band vs enhanced) and writes a comparative CSV and DuckDB table

Usage:
  python scripts/analyze_run_summary.py --run-dir <path> [--compare-dir <path>] \
      [--db results/ddmm.duckdb] [--out-dir assets/paper]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_enhanced_summary(run_dir: Path) -> Dict[str, Any]:
    # Try multiple known summary file patterns in priority order
    candidates = [
        run_dir / "run_summary_enhanced.json",
        # extracted dynesty checkpoint summaries
        *sorted(run_dir.glob("extracted*_summary*.json")),
        run_dir / "run_summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError:
                # If it's the basic run_summary.json and failed, synthesize a minimal record
                if p.name == "run_summary.json":
                    try:
                        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        raw = {}
                    return {
                        "metadata": {
                            "timestamp": raw.get("timestamp"),
                            "status": raw.get("status", "failed"),
                            "xi_type": "tidal_band",
                            "output_dir": str(run_dir),
                        },
                        "sampling_config": {
                            "nlive": raw.get("nlive"),
                            "maxcall": raw.get("maxcall"),
                            "dlogz_target": None,
                            "sample_method": None,
                            "bound_method": None,
                        },
                        "convergence_metrics": {},
                        "parameter_estimates": {},
                        "evidence_metrics": {},
                        "performance_metrics": {},
                        "model_comparison": {},
                        "quality_assessment": {"status": raw.get("status", "failed")},
                    }
                else:
                    print(f"WARNING: Could not parse JSON: {p}")
                    continue
    raise SystemExit(f"ERROR: No recognizable summary JSON in {run_dir}")


def compact_from_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    meta = s.get("metadata", {})
    conv = s.get("convergence_metrics", {})
    ev = s.get("evidence_metrics", {})
    params = s.get("parameter_estimates", {}).get("best_fit", {})
    comp = s.get("model_comparison", {}).get("vs_gr", {})
    qual = s.get("quality_assessment", {})
    return {
        "timestamp": meta.get("timestamp"),
        "xi_type": meta.get("xi_type"),
        "logz": conv.get("current_logz", ev.get("logz")),
        "logz_error": ev.get("logz_error"),
        "rho_c_solar_kpc3_best": params.get("rho_c_solar_kpc3"),
        "gamma_exp_best": params.get("gamma_exp"),
        "lambda_max_best": params.get("lambda_max"),
        "n_samples": conv.get("n_samples"),
        "efficiency": s.get("performance_metrics", {}).get("efficiency"),
        "delta_logz_vs_gr": comp.get("delta_logz"),
        "bayes_factor_log10": comp.get("bayes_factor_log10"),
        "status": qual.get("status", meta.get("status")),
    }


def write_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_records(run_name: str, comp: Dict[str, Any]) -> Dict[str, Any]:
    rec = comp.copy()
    rec["run_name"] = run_name
    return rec


def try_write_duckdb(db_path: Optional[Path], table: str, rows: List[Dict[str, Any]]):
    if not db_path:
        return
    try:
        import duckdb
        import pandas as pd
        df = pd.DataFrame(rows)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df").close()
        # Upsert-like: append then leave dedup to downstream if desired
        con = duckdb.connect(str(db_path))
        con.register("df", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
        con.close()
    except Exception as e:
        print(f"WARNING: DuckDB write failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--compare-dir", action="append", default=[], help="Optional additional run dirs to compare")
    ap.add_argument("--db", default="results/ddmm.duckdb")
    ap.add_argument("--out-dir", default="assets/paper")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    # Load main run
    main_summary = load_enhanced_summary(run_dir)
    main_compact = compact_from_summary(main_summary)

    run_name = run_dir.name
    # Write compact summary JSON under results/<run_name>
    out_summary = Path("results") / run_name / "summary_min.json"
    write_json(main_compact, out_summary)

    # Prepare comparative records
    rows = [to_records(run_name, main_compact)]

    # Optionally compare with other runs
    for cdir in args.compare_dir:
        cpath = Path(cdir)
        if not cpath.exists():
            print(f"WARNING: compare-dir missing: {cpath}")
            continue
        cs = load_enhanced_summary(cpath)
        cc = compact_from_summary(cs)
        rows.append(to_records(cpath.name, cc))
        # Also emit compact summary for each compare run
        write_json(cc, Path("results") / cpath.name / "summary_min.json")

    # Write comparative CSV
    import csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparative_evidence.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write DuckDB table
    db_path = Path(args.db) if args.db else None
    try_write_duckdb(db_path, "run_evidence", rows)

    print("Analysis complete.")
    print(f"- Main summary: {out_summary}")
    print(f"- Comparative CSV: {csv_path}")
    if db_path:
        print(f"- DuckDB: {db_path}")


if __name__ == "__main__":
    main()
