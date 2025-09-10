# tools/summarize_dmax_sweep.py
import argparse
import json
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Summarize Dmax sweep results into a single CSV")
    ap.add_argument("--roots", nargs="*", default=["results/dmax_sweep/30", "results/dmax_sweep/50", "results/dmax_sweep/80"], help="List of result roots to scan")
    ap.add_argument("--out", default="results/qa/dmax_summary.csv", help="Output CSV path")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        try:
            D = int(Path(root).name)
        except Exception:
            D = None
        p = Path(root) / "lensing_thetaE_metrics.json"
        if not p.exists():
            print(f"[warn] Missing {p}")
            continue
        m = json.loads(p.read_text())
        rows.append({
            "Dmax": D,
            "RMSE_rel": m.get("RMSE_rel"),
            "MAE_rel": m.get("MAE_rel"),
            "RMSE_abs_arcsec": m.get("RMSE_abs_arcsec"),
            "MAE_abs_arcsec": m.get("MAE_abs_arcsec"),
        })
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"Wrote {outp}")
    else:
        print("[warn] No Dmax summaries written; run the sweep first.")

if __name__ == "__main__":
    main()
