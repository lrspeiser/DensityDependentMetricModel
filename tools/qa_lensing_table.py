# tools/qa_lensing_table.py
import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="QA lensing_targets.csv for required fields")
    ap.add_argument("--in", dest="inp", default="docs/lensing_targets.csv", help="Input lens CSV path")
    ap.add_argument("--write-excluded", dest="excluded", default=None, help="Optional path to write rows with any missing required field")
    ap.add_argument("--write-clean", dest="clean", default=None, help="Optional path to write only rows with all required fields present")
    args = ap.parse_args()

    csv = Path(args.inp)
    out = Path("results/qa"); out.mkdir(parents=True, exist_ok=True)

    if not csv.exists():
        print(f"[error] Missing {csv}")
        return 1

    # Required fields for inclusion in thetaE metrics
    req = [
        "lens_id", "z_l", "z_s",
        "theta_E_obs_arcsec", "theta_E_obs_err_arcsec",
        "log10M_star", "Re_kpc", "n_sersic",
    ]
    df = pd.read_csv(csv)

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    missing = {}
    for col in req:
        if col not in df.columns:
            missing[col] = ["<column not present>"]
            continue
        mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        missing[col] = df.loc[mask, "lens_id"].astype(str).tolist() if "lens_id" in df.columns else []

    # Build a boolean report per lens
    rows = []
    idcol = "lens_id" if "lens_id" in df.columns else df.columns[0]
    for _, r in df.iterrows():
        entry = {"lens_id": str(r.get(idcol))}
        for col in req:
            val = r.get(col)
            entry[col] = bool(pd.isna(val) or str(val).strip() == "")
        rows.append(entry)

    rep = pd.DataFrame(rows).set_index("lens_id")
    rep.to_csv(out / "lensing_missing_report.csv")

    # Optional writes
    if args.excluded or args.clean:
        def is_complete(row):
            for col in req:
                if col not in row or (pd.isna(row[col]) or str(row[col]).strip() == ""):
                    return False
            return True
        mask_complete = df.apply(is_complete, axis=1)
        if args.excluded:
            Path(args.excluded).parent.mkdir(parents=True, exist_ok=True)
            df.loc[~mask_complete].to_csv(Path(args.excluded), index=False)
        if args.clean:
            Path(args.clean).parent.mkdir(parents=True, exist_ok=True)
            df.loc[mask_complete].to_csv(Path(args.clean), index=False)

    print("Missing-values by column:\n")
    for c, rows in missing.items():
        print(f"{c:>28}: {len(rows)} missing")
    print(f"\nWrote: {out/'lensing_missing_report.csv'}")
    if args.excluded:
        print(f"Wrote excluded rows: {args.excluded}")
    if args.clean:
        print(f"Wrote clean rows: {args.clean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

