import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute global alpha_lens_ph stats and residual metrics from a long-format combined lensing CSV (with run_label)."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to long-format combined lensing CSV (must include run_label, lens_id, theta_E_obs_arcsec, theta_E_RAR_phscaled_arcsec, alpha_req_at_thetaE_obs).",
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        required=False,
        default=None,
        help="Output directory for metrics, summary, and plot (defaults next to input).",
    )
    return p.parse_args()


def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def rmse(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def mae(values: List[float]) -> float:
    vals = [abs(v) for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def mape(obs: List[float], pred: List[float]) -> float:
    pairs = [(o, p) for o, p in zip(obs, pred) if math.isfinite(o) and math.isfinite(p) and o != 0.0]
    if not pairs:
        return float("nan")
    return sum(abs((p - o) / o) for o, p in pairs) / len(pairs)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    in_path = args.in_path
    out_dir = args.out_dir or os.path.join(os.path.dirname(in_path), "global_alpha")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    rows = read_csv_dicts(in_path)
    if not rows:
        raise SystemExit(f"No rows in {in_path}")

    # Collect basic columns
    need_cols = {
        "run_label",
        "lens_id",
        "theta_E_obs_arcsec",
        "theta_E_RAR_phscaled_arcsec",
        "alpha_req_at_thetaE_obs",
    }
    have = set(rows[0].keys())
    missing = [c for c in need_cols if c not in have]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Columns present: {sorted(have)}")

    # Global alpha estimates from per-lens alpha_req_at_thetaE_obs (one value per lens, independent of run)
    by_lens_alpha = {}
    for r in rows:
        lens = r.get("lens_id", "")
        a_req = safe_float(r.get("alpha_req_at_thetaE_obs", ""))
        if math.isfinite(a_req):
            by_lens_alpha[lens] = a_req  # overwrite consistently; they should match across runs

    alpha_vals = list(by_lens_alpha.values())
    alpha_stats = {
        "alpha_req_mean": mean(alpha_vals) if alpha_vals else float("nan"),
        "alpha_req_median": median(alpha_vals) if alpha_vals else float("nan"),
        "alpha_req_min": min(alpha_vals) if alpha_vals else float("nan"),
        "alpha_req_max": max(alpha_vals) if alpha_vals else float("nan"),
        "alpha_req_count": len(alpha_vals),
        "alpha_req_by_lens": by_lens_alpha,
    }

    # Residual metrics per run_label using scaled predictions
    by_run = {}
    for r in rows:
        lab = r.get("run_label", "")
        obs = safe_float(r.get("theta_E_obs_arcsec", ""))
        pred = safe_float(r.get("theta_E_RAR_phscaled_arcsec", ""))
        if lab not in by_run:
            by_run[lab] = {"obs": [], "pred": [], "lens": []}
        by_run[lab]["obs"].append(obs)
        by_run[lab]["pred"].append(pred)
        by_run[lab]["lens"].append(r.get("lens_id", ""))

    metrics_rows: List[Dict[str, str]] = []
    for lab, d in sorted(by_run.items()):
        obs = d["obs"]
        pred = d["pred"]
        residuals = [p - o for p, o in zip(pred, obs)]
        metrics_rows.append(
            {
                "run_label": lab,
                "N": str(len(obs)),
                "MAE_arcsec": f"{mae(residuals):.6g}",
                "RMSE_arcsec": f"{rmse(residuals):.6g}",
                "MAPE": f"{mape(obs, pred):.6g}",
            }
        )

    metrics_path = os.path.join(out_dir, "lensing_global_alpha_metrics.csv")
    write_csv(metrics_path, ["run_label", "N", "MAE_arcsec", "RMSE_arcsec", "MAPE"], metrics_rows)

    # Write summary JSON for alpha
    summary_path = os.path.join(out_dir, "lensing_global_alpha_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(alpha_stats, f, indent=2)

    # Plot predicted vs observed by run (scaled predictions)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
        # Build a combined scatter; color by run
        labels = sorted(by_run.keys())
        colors = ["#f39c12", "#e67e22", "#e84393", "#9b59b6", "#16a085", "#2ecc71", "#c0392b"]
        for i, lab in enumerate(labels):
            d = by_run[lab]
            ax.scatter(d["obs"], d["pred"], s=32, marker="x", color=colors[i % len(colors)], label=lab)
        # y=x reference
        all_obs = [safe_float(r.get("theta_E_obs_arcsec", "")) for r in rows]
        x0, x1 = min(all_obs or [0.0]), max(all_obs or [1.0])
        ax.plot([0, x1], [0, x1], ls="--", lw=1.5, color="#f1c40f")
        ax.set_xlabel("Observed theta_E [arcsec]")
        ax.set_ylabel("Predicted theta_E (scaled) [arcsec]")
        ax.set_title("Einstein radius: predicted vs observed (by run)")
        ax.legend(frameon=False, fontsize=8)
        plot_path = os.path.join(out_dir, "lensing_global_alpha_pred_vs_obs.png")
        fig.savefig(plot_path, dpi=150)

    print(f"[OK] Wrote metrics: {metrics_path}")
    print(f"[OK] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
