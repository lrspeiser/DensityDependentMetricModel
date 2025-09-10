# tools/compare_lensing_systematics.py
import argparse
import json
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Compare baseline vs systematics lensing outputs")
    ap.add_argument("--base", default="results/next_steps/enhanced_20250805_115400", help="Baseline results root")
    ap.add_argument("--sys", dest="sysroot", default="results/next_steps_sys/enhanced_20250805_115400", help="Systematics results root")
    ap.add_argument("--out-dir", default="results/qa", help="Output directory for CSVs")
    args = ap.parse_args()

    base = Path(args.base)
    sysr = Path(args.sysroot)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # θE metrics comparison
    mb_p = base / "lensing_thetaE_metrics.json"
    ms_p = sysr / "lensing_thetaE_metrics.json"
    if mb_p.exists() and ms_p.exists():
        mb = json.loads(mb_p.read_text())
        ms = json.loads(ms_p.read_text())
        def pick(m):
            return {k:v for k,v in m.items() if any(s in k for s in ["RMSE","MAE","Bias"]) or k=="N"}
        theta_summary = pd.DataFrame([
            {"run":"baseline", **pick(mb)},
            {"run":"systematics", **pick(ms)},
        ])
        theta_summary.to_csv(out/"thetaE_metrics_comparison.csv", index=False)
        print(f"[ok] wrote {out/'thetaE_metrics_comparison.csv'}")
    else:
        print("[warn] Missing one of the thetaE metrics JSONs; skipping θE comparison.")

    # Stacked ΔΣ deltas at representative radii
    cb_p = base / "lensing_metric_stack.csv"
    cs_p = sysr / "lensing_metric_stack.csv"
    if cb_p.exists() and cs_p.exists():
        cb = pd.read_csv(cb_p)
        cs = pd.read_csv(cs_p)
        at = [10, 30, 100]  # kpc
        def interp(df, r):
            i = (df["R_kpc"]-r).abs().argmin()
            return float(df.iloc[i]["DeltaSigma_mean"])
        stack_summary = pd.DataFrame({
            "R_kpc": at,
            "DeltaSigma_mean_baseline": [interp(cb,r) for r in at],
            "DeltaSigma_mean_sys":      [interp(cs,r) for r in at],
        })
        stack_summary["delta_frac"] = (stack_summary["DeltaSigma_mean_sys"] / stack_summary["DeltaSigma_mean_baseline"]) - 1.0
        stack_summary.to_csv(out/"stack_deltas.csv", index=False)
        print(f"[ok] wrote {out/'stack_deltas.csv'}")
    else:
        print("[warn] Missing one of the stack CSVs; skipping stack delta summary.")

if __name__ == "__main__":
    main()
