# tools/summarize_dmax_sweep.py
import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_dmax(root: str):
    name = Path(root).name
    try:
        return float(name), name
    except Exception:
        # allow labels like 'inf'
        return None, name


def _read_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Summarize Dmax sweep results into a single CSV")
    ap.add_argument(
        "--roots",
        nargs="*",
        default=[
            "results/dmax_sweep/30",
            "results/dmax_sweep/50",
            "results/dmax_sweep/80",
            "results/dmax_sweep/inf",
        ],
        help="List of result roots to scan",
    )
    ap.add_argument("--out", default="results/qa/dmax_summary.csv", help="Output CSV path")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        rootp = Path(root)
        dmax_num, dmax_label = _parse_dmax(root)

        # Lensing metrics (optional)
        lens_json = rootp / "lensing_thetaE_metrics.json"
        lens = _read_json(lens_json) or {}

        # SPARC per-galaxy summary (optional)
        sparc_csv = rootp / "sparc_a0_summary.csv"
        n_sparc = 0
        red_rar = red_gr = med_dchi2 = frac_better = None
        try:
            if sparc_csv.exists():
                sdf = pd.read_csv(sparc_csv)
                # Expect columns: chi2_rar, chi2_gr, dof
                if set(["chi2_rar", "chi2_gr", "dof"]).issubset(sdf.columns):
                    n_sparc = int(len(sdf))
                    tot_dof = float(sdf["dof"].sum()) if n_sparc > 0 else 0.0
                    sum_rar = float(sdf["chi2_rar"].sum())
                    sum_gr = float(sdf["chi2_gr"].sum())
                    red_rar = (sum_rar / tot_dof) if tot_dof > 0 else None
                    red_gr = (sum_gr / tot_dof) if tot_dof > 0 else None
                    dchi2 = (sdf["chi2_gr"] - sdf["chi2_rar"]).astype(float)
                    med_dchi2 = float(dchi2.median()) if len(dchi2) > 0 else None
                    frac_better = float((sdf["chi2_rar"] < sdf["chi2_gr"]).mean()) if n_sparc > 0 else None
        except Exception:
            pass

        # Model comparison (optional)
        mc_csv = rootp / "model_comparison_bic.csv"
        med_dlogZ = None
        try:
            if mc_csv.exists():
                mdf = pd.read_csv(mc_csv)
                if "delta_logZ_rar_vs_gr" in mdf.columns and len(mdf) > 0:
                    med_dlogZ = float(pd.to_numeric(mdf["delta_logZ_rar_vs_gr"], errors="coerce").median())
        except Exception:
            pass

        # MW Kz (optional): value at ~1.1 kpc
        kz_csv = rootp / "mw_kz_sigma_full3d.csv"
        kz_1p1 = None
        try:
            if kz_csv.exists():
                kdf = pd.read_csv(kz_csv)
                if set(["z_kpc", "Kz_m_s2"]).issubset(kdf.columns) and len(kdf) > 0:
                    # nearest to 1.1 kpc
                    idx = (kdf["z_kpc"] - 1.1).abs().idxmin()
                    kz_1p1 = float(kdf.loc[idx, "Kz_m_s2"]) if idx is not None else None
        except Exception:
            pass

        rows.append({
            "Dmax_label": dmax_label,
            "Dmax_numeric": dmax_num,
            # Lensing
            "N_lens": lens.get("N"),
            "RMSE_rel": lens.get("RMSE_rel"),
            "MAE_rel": lens.get("MAE_rel"),
            "RMSE_abs_arcsec": lens.get("RMSE_abs_arcsec"),
            "MAE_abs_arcsec": lens.get("MAE_abs_arcsec"),
            # SPARC
            "N_sparc": n_sparc,
            "sparc_red_chi2_rar": red_rar,
            "sparc_red_chi2_gr": red_gr,
            "sparc_median_delta_chi2": med_dchi2,
            "sparc_frac_rar_better": frac_better,
            # Model comparison
            "median_delta_logZ_rar_vs_gr": med_dlogZ,
            # MW Kz
            "Kz_1p1_m_s2": kz_1p1,
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
