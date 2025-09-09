# tools/summarize_dmax_sweep.py
import json
import pandas as pd
from pathlib import Path

def main():
    roots = [30, 50, 80]
    rows = []
    for D in roots:
        p = Path(f"results/dmax_sweep/{D}/lensing_thetaE_metrics.json")
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
    out = Path("results/qa"); out.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(out/"dmax_summary.csv", index=False)
        print(f"Wrote {out/'dmax_summary.csv'}")
    else:
        print("[warn] No Dmax summaries written; run the sweep first.")

if __name__ == "__main__":
    main()

