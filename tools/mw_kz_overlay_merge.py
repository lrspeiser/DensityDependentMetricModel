# tools/mw_kz_overlay_merge.py
# Create a unified overlay CSV with two bands so both can be shown together.
import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Merge two MW Kz overlay CSVs (Bovy–Rix + McMillan)")
    ap.add_argument("--in1", default="docs/mw_kz_overlay_bovyrix2013_SCALED.csv", help="First overlay CSV (e.g., Bovy–Rix)")
    ap.add_argument("--in2", default="docs/mw_kz_overlay_mcmillan2022_SCALED.csv", help="Second overlay CSV (e.g., McMillan)")
    ap.add_argument("--out", default="docs/mw_kz_overlay_2band.csv", help="Output merged CSV path")
    args = ap.parse_args()

    br = Path(args.in1)
    mc = Path(args.in2)
    if not br.exists():
        print(f"[error] Missing {br}")
        return 1
    if not mc.exists():
        print(f"[error] Missing {mc}")
        return 1
    dbr = pd.read_csv(br)
    dmc = pd.read_csv(mc)
    merged = dbr.merge(dmc, on="z_kpc", how="outer", suffixes=("_BR13","_MC22")).sort_values("z_kpc")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
