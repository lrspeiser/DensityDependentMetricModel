#!/usr/bin/env python3
"""
Analyze a completed run directory (e.g., runs/nfw_YYYYMMDD_HHMMSS):
- Load posterior_samples.npz
- Compute weighted parameter summaries (mean, std, median, 68% and 95% intervals)
- Save JSON and CSV summaries
- Generate basic plots (marginal histograms, pair plot, logL histogram)
Artifacts saved under <run_dir>/analysis

Usage:
  python scripts/analyze_run.py --run-dir runs/nfw_20250812_114008 --label NFW --overwrite

This script is self-contained and does not require project modules that may fail to import during CI.
"""
import argparse
import os
import json
import csv
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    """Compute weighted quantiles of 1D values.
    values: shape (n,)
    quantiles: array of probs in [0,1]
    sample_weight: shape (n,)
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)
    assert values.ndim == 1
    assert sample_weight.ndim == 1
    assert values.shape[0] == sample_weight.shape[0]
    if not np.all(np.isfinite(values)):
        mask = np.isfinite(values)
        values = values[mask]
        sample_weight = sample_weight[mask]
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    if cdf[-1] <= 0:
        # fallback to uniform
        sample_weight = np.ones_like(values) / len(values)
        cdf = np.cumsum(sample_weight)
    cdf = (cdf - 0.5 * sample_weight) / cdf[-1]
    return np.interp(quantiles, cdf, values)


def load_posterior(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    data = np.load(npz_path, allow_pickle=True)
    # Try common key names
    samples = None
    weights = None
    logl = None
    names = None
    for k in ("samples", "xs", "points", "theta"):
        if k in data.files:
            samples = data[k]
            break
    for k in ("weights", "ws", "w"):
        if k in data.files:
            weights = data[k]
            break
    for k in ("logl", "logl_values", "logL", "loglike"):
        if k in data.files:
            logl = data[k]
            break
    if "names" in data.files:
        n = data["names"]
        try:
            names = list(n.tolist())
        except Exception:
            names = [str(x) for x in n]
    # Fallback parameter names
    if samples is None:
        raise RuntimeError(f"No samples array found in {npz_path}. Keys: {list(data.files)}")
    if names is None:
        names = [f"param_{i}" for i in range(samples.shape[1])]
    if weights is None:
        # create uniform weights if missing
        weights = np.ones(samples.shape[0], dtype=float) / samples.shape[0]
    else:
        # normalize weights
        ws = np.asarray(weights, dtype=float)
        s = ws.sum()
        if s <= 0 or not np.isfinite(s):
            ws = np.ones_like(ws) / len(ws)
        else:
            ws = ws / s
        weights = ws
    return samples, weights, logl, names


def summarize_posterior(samples: np.ndarray, weights: np.ndarray, names: List[str]) -> dict:
    quant_probs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    summary = {}
    for j, name in enumerate(names):
        v = samples[:, j]
        q = weighted_quantile(v, quant_probs, weights)
        mean = np.average(v, weights=weights)
        var = np.average((v - mean) ** 2, weights=weights)
        std = np.sqrt(max(var, 0.0))
        summary[name] = {
            "mean": float(mean),
            "std": float(std),
            "q2.5": float(q[0]),
            "q16": float(q[1]),
            "median": float(q[2]),
            "q84": float(q[3]),
            "q97.5": float(q[4]),
        }
    return summary


def write_summaries(analysis_dir: Path, summary: dict, samples: np.ndarray, weights: np.ndarray, names: List[str], meta: dict):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    out_json = analysis_dir / "posterior_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "parameters": summary}, f, indent=2)
    # CSV
    out_csv = analysis_dir / "posterior_summary.csv"
    fieldnames = ["name", "mean", "std", "q2.5", "q16", "median", "q84", "q97.5"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, stats in summary.items():
            row = {"name": name}
            row.update({k: stats[k] for k in fieldnames if k != "name"})
            w.writerow(row)


def plot_marginals(analysis_dir: Path, samples: np.ndarray, weights: np.ndarray, names: List[str], label: str):
    npar = samples.shape[1]
    fig, axes = plt.subplots(npar, 1, figsize=(6, max(2.0, 1.8 * npar)), constrained_layout=True)
    if npar == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        v = samples[:, j]
        ax.hist(v, bins=50, weights=weights, color="#4477aa", alpha=0.8)
        ax.set_xlabel(names[j])
        ax.set_ylabel("weighted density")
    fig.suptitle(f"Marginal posteriors: {label}")
    fig.savefig(analysis_dir / "marginals.png", dpi=160)
    plt.close(fig)


def plot_pair(analysis_dir: Path, samples: np.ndarray, weights: np.ndarray, names: List[str], label: str):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d), constrained_layout=True)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, j], bins=40, weights=weights, color="#4477aa", alpha=0.8)
                ax.set_ylabel(names[j])
            else:
                ax.hexbin(samples[:, j], samples[:, i], C=weights, gridsize=40, cmap="viridis")
            if i == d - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
    fig.suptitle(f"Pair plot: {label}")
    fig.savefig(analysis_dir / "pair.png", dpi=160)
    plt.close(fig)


def plot_logl(analysis_dir: Path, logl: np.ndarray, label: str):
    if logl is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.hist(logl, bins=60, color="#aa7744", alpha=0.8)
    ax.set_xlabel("log-likelihood")
    ax.set_ylabel("count")
    ax.set_title(f"LogL distribution: {label}")
    fig.savefig(analysis_dir / "logl_hist.png", dpi=160)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to completed run directory")
    ap.add_argument("--label", default="Run", help="Label for plots")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing analysis")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(2)
    npz_path = run_dir / "posterior_samples.npz"
    if not npz_path.exists():
        print(f"posterior_samples.npz not found in {run_dir}", file=sys.stderr)
        sys.exit(2)
    analysis_dir = run_dir / "analysis"
    if analysis_dir.exists() and not args.overwrite:
        print(f"Analysis directory exists: {analysis_dir}. Use --overwrite to refresh.")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    samples, weights, logl, names = load_posterior(npz_path)
    meta = {
        "run_dir": str(run_dir),
        "n_samples": int(samples.shape[0]),
        "n_params": int(samples.shape[1]),
        "names": names,
        "label": args.label,
    }
    # Try to incorporate run summary if present
    summary_path = run_dir / "run_summary_enhanced.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                run_summary = json.load(f)
            meta["logz"] = run_summary.get("Convergence Metrics", {}).get("Current LogZ")
            meta["status"] = run_summary.get("Quality Assessment", {}).get("Status")
            meta["xi_type"] = run_summary.get("Run Information", {}).get("Xi Type")
        except Exception:
            pass

    # Summaries
    param_summary = summarize_posterior(samples, weights, names)
    write_summaries(analysis_dir, param_summary, samples, weights, names, meta)

    # Plots
    plot_marginals(analysis_dir, samples, weights, names, args.label)
    if samples.shape[1] >= 2:
        plot_pair(analysis_dir, samples, weights, names, args.label)
    plot_logl(analysis_dir, logl, args.label)

    # Lightweight text report
    with (analysis_dir / "REPORT.txt").open("w", encoding="utf-8") as f:
        f.write(f"Analysis for {args.label}\n")
        f.write(json.dumps(meta, indent=2))
        f.write("\n\nParameter summaries (median [16%, 84%]):\n")
        for name, stats in param_summary.items():
            f.write(f"- {name}: {stats['median']:.5g} [{stats['q16']:.5g}, {stats['q84']:.5g}]\n")
    print(f"Saved analysis artifacts to {analysis_dir}")


if __name__ == "__main__":
    main()

