# tools/mw_kz_overlay_merge.py
# Create a unified overlay CSV with two bands so both can be shown together.
import pandas as pd
from pathlib import Path

def main():
    br = Path("docs/mw_kz_overlay_bovyrix2013_SCALED.csv")
    mc = Path("docs/mw_kz_overlay_mcmillan2022_SCALED.csv")
    if not br.exists():
        print(f"[error] Missing {br}")
        return 1
    if not mc.exists():
        print(f"[error] Missing {mc}")
        return 1
    dbr = pd.read_csv(br)
    dmc = pd.read_csv(mc)
    merged = dbr.merge(dmc, on="z_kpc", how="outer", suffixes=("_BR13","_MC22")).sort_values("z_kpc")
    out = Path("docs/mw_kz_overlay_2band.csv")
    merged.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

