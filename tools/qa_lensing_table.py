# tools/qa_lensing_table.py
import pandas as pd
from pathlib import Path


def main():
    csv = Path("docs/lensing_targets.csv")
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

    print("Missing-values by column:\n")
    for c, rows in missing.items():
        print(f"{c:>28}: {len(rows)} missing")
    print(f"\nWrote: {out/'lensing_missing_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

