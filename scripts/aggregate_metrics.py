#!/usr/bin/env python3
# Aggregates lensing metrics across runs and generates RMS plots.
# Outputs combined CSVs and figures for:
# - alpha-only sweeps (zeta=0, env=constant) per profile
# - tapered zeta sweeps at alpha=2.0 per profile

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_lensing_table(csv_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header_map: Dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, name in enumerate(header):
            header_map[name] = idx
        for parts in reader:
            if not parts:
                continue
            row = {name: parts[idx] for name, idx in header_map.items()}
            rows.append(row)
    return rows, header_map


def to_float(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_metrics_for_run(run_dir: Path) -> Dict[str, float]:
    """Compute basic metrics for a single run directory containing lensing_rar_table.csv.

    Returns dict with alpha, zeta, env_profile, profile, n, RMS_abs, RMS_rel, median_rel, mean_rel, max_rel.
    """
    rar_csv = run_dir / "lensing_rar_table.csv"
    if not rar_csv.exists():
        raise FileNotFoundError(f"Missing lensing_rar_table.csv in {run_dir}")

    rows, header_map = read_lensing_table(rar_csv)
    n = 0
    diffs_abs: List[float] = []
    diffs_rel: List[float] = []

    # Infer run parameters from CSV content
    # Columns we saw in sample: theta_E_obs_arcsec, theta_E_RAR_phscaled_arcsec, alpha_lens_ph_used, zeta_env_lens_used, env_profile
    # Profile isn't stored; derive from directory name suffix
    alpha = None
    zeta = None
    env_profile = None

    for row in rows:
        obs = to_float(row.get("theta_E_obs_arcsec", "nan"))
        scaled = to_float(row.get("theta_E_RAR_phscaled_arcsec", "nan"))
        if math.isfinite(obs) and math.isfinite(scaled) and obs != 0.0:
            diffs_abs.append(scaled - obs)
            diffs_rel.append((scaled - obs) / obs)
            n += 1
        # Capture params (same across rows)
        if alpha is None:
            alpha = to_float(row.get("alpha_lens_ph_used", "nan"))
        if zeta is None:
            zeta = to_float(row.get("zeta_env_lens_used", "nan"))
        if env_profile is None:
            env_profile = row.get("env_profile", None)

    if n == 0:
        raise ValueError(f"No usable lens rows in {rar_csv}")

    rms_abs = math.sqrt(sum(d*d for d in diffs_abs) / n)
    rms_rel = math.sqrt(sum(d*d for d in diffs_rel) / n)

    diffs_rel_sorted = sorted(diffs_rel)
    median_rel = diffs_rel_sorted[n // 2] if n % 2 == 1 else 0.5 * (diffs_rel_sorted[n // 2 - 1] + diffs_rel_sorted[n // 2])
    mean_rel = sum(diffs_rel) / n
    max_rel = max(abs(d) for d in diffs_rel)

    # Infer profile from directory name
    name = run_dir.name
    profile = "hernquist" if name.endswith("_hernquist") else ("jaffe" if name.endswith("_jaffe") else "unknown")

    # Determine category
    # alpha-only: env_profile == constant and (zeta==0 or close)
    # zeta-scan: env_profile == tapered (we'll filter by alpha ~ 2.0 when writing tables)
    try:
        is_alpha_only = (env_profile == "constant") and (abs((zeta or 0.0) - 0.0) < 1e-9)
    except Exception:
        is_alpha_only = False

    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "profile": profile,
        "alpha": float(alpha if alpha is not None else float("nan")),
        "zeta": float(zeta if zeta is not None else float("nan")),
        "env_profile": env_profile or "",
        "is_alpha_only": 1.0 if is_alpha_only else 0.0,
        "n": float(n),
        "rms_abs": rms_abs,
        "rms_rel": rms_rel,
        "median_rel": median_rel,
        "mean_rel": mean_rel,
        "max_rel": max_rel,
    }


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def plot_rms(x_vals: List[float], y_vals: List[float], xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(x_vals, y_vals, marker='o')
    # best point
    if y_vals:
        best_idx = min(range(len(y_vals)), key=lambda i: y_vals[i])
        plt.scatter([x_vals[best_idx]], [y_vals[best_idx]], color='red', zorder=3, label=f"best={x_vals[best_idx]:g}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="results/next_steps/btfr_fix_20250906_lastcross")
    ap.add_argument("--out-root", default="results/next_steps/btfr_fix_20250906_lastcross/combined_metrics")
    ap.add_argument("--images-root", default="images/next_steps/btfr_fix_20250906_lastcross/metrics")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_root = Path(args.out_root)
    img_root = Path(args.images_root)

    # Collect all runs with lensing_rar_table.csv
    runs: List[Dict[str, float]] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        rar = child / "lensing_rar_table.csv"
        if rar.exists():
            try:
                metrics = compute_metrics_for_run(child)
                runs.append(metrics)
            except Exception as e:
                print(f"[WARN] Skipping {child}: {e}")

    if not runs:
        raise SystemExit("No runs with lensing_rar_table.csv found.")

    # Write master metrics table
    fields = [
        "run_dir","profile","alpha","zeta","env_profile","is_alpha_only","n",
        "rms_abs","rms_rel","median_rel","mean_rel","max_rel"
    ]
    write_csv(out_root / "metrics_all_runs.csv", runs, fields)

    # Alpha-only per profile (env constant, zeta=0)
    for profile in ("hernquist", "jaffe"):
        alpha_runs = [r for r in runs if r["profile"] == profile and r["is_alpha_only"] == 1.0]
        alpha_runs_sorted = sorted(alpha_runs, key=lambda r: r["alpha"])
        write_csv(out_root / f"metrics_alpha_only_{profile}.csv", alpha_runs_sorted, fields)
        if alpha_runs_sorted:
            x = [r["alpha"] for r in alpha_runs_sorted]
            y = [r["rms_rel"] for r in alpha_runs_sorted]
            plot_rms(x, y, xlabel="alpha", ylabel="RMS relative error", title=f"RMS_rel vs alpha ({profile})", out_path=img_root / f"rms_rel_vs_alpha_{profile}.png")

    # Zeta sweeps at alpha=2.0 per profile (env tapered)
    for profile in ("hernquist", "jaffe"):
        zeta_runs = [r for r in runs if r["profile"] == profile and r["env_profile"] == "tapered" and abs(r["alpha"] - 2.0) < 1e-6]
        zeta_runs_sorted = sorted(zeta_runs, key=lambda r: r["zeta"])
        write_csv(out_root / f"metrics_zeta_alpha2_{profile}.csv", zeta_runs_sorted, fields)
        if zeta_runs_sorted:
            x = [r["zeta"] for r in zeta_runs_sorted]
            y = [r["rms_rel"] for r in zeta_runs_sorted]
            plot_rms(x, y, xlabel="zeta", ylabel="RMS relative error", title=f"RMS_rel vs zeta @ alpha=2.0 ({profile})", out_path=img_root / f"rms_rel_vs_zeta_alpha2_{profile}.png")

    print("[INFO] Aggregation complete.")
    print(f"[INFO] Wrote {out_root / 'metrics_all_runs.csv'}")
    print(f"[INFO] Images in {img_root}")


if __name__ == "__main__":
    main()

