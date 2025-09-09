# tools/sparc_ppc.py
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    res_path = Path("results/next_steps/enhanced_20250805_115400/sparc_residuals.csv")
    out = Path("results/qa"); out.mkdir(parents=True, exist_ok=True)

    if not res_path.exists():
        print(f"[warn] Missing {res_path}; provide a per-point residuals CSV with columns: galaxy,R_kpc,v_obs,v_mod,sigma_v")
        return 1

    df = pd.read_csv(res_path)
    for col in ["v_obs","v_mod","sigma_v"]:
        if col not in df.columns:
            print(f"[error] Column {col} not found in {res_path}")
            return 1
    df["z"] = (df["v_obs"] - df["v_mod"]) / df["sigma_v"].replace(0, np.nan)

    # Per-galaxy summary
    summary = df.groupby("galaxy")["z"].agg(["count","mean","std"]).reset_index()
    summary.to_csv(out/"sparc_ppc_summary.csv", index=False)

    # Histogram data for SI plot
    hist, edges = np.histogram(df["z"].clip(-5,5).dropna(), bins=30, density=True)
    pd.DataFrame({"bin_lo":edges[:-1], "bin_hi":edges[1:], "density":hist}).to_csv(out/"sparc_ppc_hist.csv", index=False)

    print("Wrote PPC summaries in", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

