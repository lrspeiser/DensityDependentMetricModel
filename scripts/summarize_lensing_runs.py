#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def parse_runs(run_args):
    runs = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Run spec must be name=path, got: {item}")
        name, path = item.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Run '{name}' CSV not found: {p}")
        runs[name] = p
    return runs


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def to_float(row, key):
    val = row.get(key, None)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Summarize RAR lensing runs across CSVs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of name=path entries for lensing_rar_table.csv files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/next_steps/lensing_summary.csv",
        help="Output CSV path for the summary",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)

    # Gather data keyed by lens_id
    per_lens = defaultdict(dict)
    meta_keys = [
        "z_l",
        "z_s",
        "Re_kpc",
        "log10M_star",
        "n_sersic",
        "theta_E_obs_arcsec",
        "theta_E_GR_arcsec",
        "theta_E_SIS200_arcsec",
        "theta_E_SIS250_arcsec",
    ]

    for name, path in runs.items():
        rows = read_csv(path)
        for row in rows:
            lens_id = row["lens_id"]
            # Store shared meta from first occurrence
            if "meta" not in per_lens[lens_id]:
                per_lens[lens_id]["meta"] = {
                    k: row.get(k) for k in meta_keys if k in row
                }
            # Store per-run predictions
            per_lens[lens_id][name] = {
                "alpha_lens_ph_used": row.get("alpha_lens_ph_used"),
                "zeta_env_lens_used": row.get("zeta_env_lens_used"),
                "env_profile": row.get("env_profile"),
                "theta_E_RAR_arcsec": row.get("theta_E_RAR_arcsec"),
                # scaled prediction is the effective one used for comparison
                "theta_E_RAR_phscaled_arcsec": row.get("theta_E_RAR_phscaled_arcsec"),
                "alpha_req_at_thetaE_obs": row.get("alpha_req_at_thetaE_obs"),
            }

    # Compose output headers
    out_cols = [
        "lens_id",
        "z_l",
        "z_s",
        "theta_E_obs_arcsec",
        "theta_E_GR_arcsec",
        "theta_E_SIS200_arcsec",
        "theta_E_SIS250_arcsec",
    ]

    # Add per-run columns
    for name in runs.keys():
        out_cols.extend([
            f"{name}__alpha_lens_ph_used",
            f"{name}__zeta_env_lens_used",
            f"{name}__env_profile",
            f"{name}__theta_E_RAR_phscaled_arcsec",
            f"{name}__resid_arcsec",
            f"{name}__resid_pct",
            f"{name}__alpha_req_at_thetaE_obs",
        ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for lens_id, bundle in per_lens.items():
            meta = bundle.get("meta", {})
            row_out = {
                "lens_id": lens_id,
                "z_l": meta.get("z_l"),
                "z_s": meta.get("z_s"),
                "theta_E_obs_arcsec": meta.get("theta_E_obs_arcsec"),
                "theta_E_GR_arcsec": meta.get("theta_E_GR_arcsec"),
                "theta_E_SIS200_arcsec": meta.get("theta_E_SIS200_arcsec"),
                "theta_E_SIS250_arcsec": meta.get("theta_E_SIS250_arcsec"),
            }
            obs = None
            try:
                obs = float(meta.get("theta_E_obs_arcsec")) if meta.get("theta_E_obs_arcsec") is not None else None
            except Exception:
                obs = None
            for name in runs.keys():
                dat = bundle.get(name, {})
                pred = to_float(dat, "theta_E_RAR_phscaled_arcsec")
                resid = pred - obs if (pred is not None and obs is not None) else None
                resid_pct = (pred / obs - 1.0) * 100.0 if (pred is not None and obs is not None and obs != 0.0) else None
                row_out[f"{name}__alpha_lens_ph_used"] = dat.get("alpha_lens_ph_used")
                row_out[f"{name}__zeta_env_lens_used"] = dat.get("zeta_env_lens_used")
                row_out[f"{name}__env_profile"] = dat.get("env_profile")
                row_out[f"{name}__theta_E_RAR_phscaled_arcsec"] = dat.get("theta_E_RAR_phscaled_arcsec")
                row_out[f"{name}__resid_arcsec"] = f"{resid:.6f}" if resid is not None else ""
                row_out[f"{name}__resid_pct"] = f"{resid_pct:.2f}" if resid_pct is not None else ""
                row_out[f"{name}__alpha_req_at_thetaE_obs"] = dat.get("alpha_req_at_thetaE_obs")
            writer.writerow(row_out)

    # Print concise table to stdout
    print("Summary (arcsec):")
    header = ["lens_id", "obs"] + [f"{name}_pred" for name in runs.keys()] + [f"{name}_resid" for name in runs.keys()]
    print(",".join(header))
    for lens_id, bundle in per_lens.items():
        meta = bundle.get("meta", {})
        obs = to_float(meta, "theta_E_obs_arcsec")
        row = [lens_id, f"{obs:.3f}" if obs is not None else ""]
        preds = []
        resids = []
        for name in runs.keys():
            dat = bundle.get(name, {})
            pred = to_float(dat, "theta_E_RAR_phscaled_arcsec")
            preds.append(f"{pred:.6f}" if pred is not None else "")
            resid = (pred - obs) if (pred is not None and obs is not None) else None
            resids.append(f"{resid:.6f}" if resid is not None else "")
        print(",".join(row + preds + resids))

    print(f"\nWrote summary CSV to: {out_path}")


if __name__ == "__main__":
    main()
