#!/usr/bin/env python3
"""
Post-analysis for dynesty runs.

Reads a run directory containing at least results.pkl and posterior_samples.npz,
computes diagnostics (normalized posterior weights, ESS), checks evidence
consistency, and produces plots and a markdown + JSON report suitable for paper
supplemental material.

Usage (PowerShell/CMD/bash):
  python scripts/analyze_results.py --run_dir runs/<run_name>

Outputs are saved under <run_dir>/post_analysis/.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class EvidenceSummary:
    logz_final: Optional[float]
    logzerr_final: Optional[float]
    n_logz_points: int


@dataclass
class PosteriorSummary:
    n_samples: int
    ess: float
    ess_ratio: float
    has_nan_weights: bool
    has_neg_weights: bool
    weight_min: float
    weight_max: float


@dataclass
class RunSummary:
    run_dir: str
    created_at: str
    evidence: EvidenceSummary
    posterior: PosteriorSummary
    params: Dict[str, Dict[str, float]]  # name -> {mean, std, p16, p50, p84}
    notes: str


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x).ravel()
            if arr.size == 0:
                return None
            return float(arr[-1])
        return float(x)
    except Exception:
        return None


def _load_results(results_path: str) -> Dict[str, Any]:
    with open(results_path, "rb") as f:
        obj = pickle.load(f)
    # dynesty can pickle a `Results` object or a dict
    if hasattr(obj, "__dict__") and (hasattr(obj, "samples") or hasattr(obj, "logz")):
        # Convert to dict of arrays where possible
        out: Dict[str, Any] = {}
        for k in dir(obj):
            if k.startswith("__"):
                continue
            try:
                v = getattr(obj, k)
            except Exception:
                continue
            if isinstance(v, (np.ndarray, list, tuple, float, int)):
                out[k] = v
        return out
    elif isinstance(obj, dict):
        return obj
    else:
        # Fallback to dict of attrs
        out: Dict[str, Any] = {}
        for k in dir(obj):
            if k.startswith("__"):
                continue
            try:
                out[k] = getattr(obj, k)
            except Exception:
                pass
        return out


def _normalize_weights_from_logwt(logwt: np.ndarray) -> np.ndarray:
    # Stable normalization: w = exp(logwt - logsumexp(logwt))
    logwt = np.asarray(logwt, dtype=float)
    m = np.max(logwt)
    lse = m + math.log(np.sum(np.exp(logwt - m)))
    w = np.exp(logwt - lse)
    # defensively renormalize
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(w)
    return w / s


def _weighted_stats_1d(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.sum(w)
    if s <= 0:
        return float("nan"), float("nan")
    mu = np.sum(w * x) / s
    var = np.sum(w * (x - mu) ** 2) / s
    var = max(var, 0.0)
    return mu, math.sqrt(var)


def _quantiles_1d(x: np.ndarray, w: np.ndarray, qs=(0.16, 0.5, 0.84)) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.sum(w)
    if s <= 0:
        return np.array([np.nan] * len(qs))
    # Weighted quantiles via sort and cumulative weights
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    cdf /= cdf[-1]
    return np.interp(qs, cdf, xs)


def analyze_run(run_dir: str) -> RunSummary:
    os.makedirs(os.path.join(run_dir, "post_analysis"), exist_ok=True)

    results_path = os.path.join(run_dir, "results.pkl")
    post_path = os.path.join(run_dir, "posterior_samples.npz")

    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Missing results.pkl at {results_path}")

    results = _load_results(results_path)

    # Evidence
    logz = np.asarray(results.get("logz", []))
    logzerr = np.asarray(results.get("logzerr", []))
    ev = EvidenceSummary(
        logz_final=_safe_float(logz[-1] if logz.size > 0 else None),
        logzerr_final=_safe_float(logzerr[-1] if logzerr.size > 0 else None),
        n_logz_points=int(logz.size),
    )

    # Posterior weights and ESS
    if "logwt" in results:
        logwt = np.asarray(results["logwt"], dtype=float)
        weights = _normalize_weights_from_logwt(logwt)
    elif "weights" in results:
        weights = np.asarray(results["weights"], dtype=float)
        s = np.sum(weights)
        weights = weights / s if s > 0 else np.zeros_like(weights)
    else:
        weights = np.array([])

    has_nan = bool(np.any(~np.isfinite(weights))) if weights.size else False
    has_neg = bool(np.any(weights < 0)) if weights.size else False
    wsum = float(np.sum(weights)) if weights.size else 0.0
    if weights.size and (not np.isfinite(wsum) or abs(wsum - 1.0) > 1e-6):
        # Renormalize if needed
        s = np.sum(weights)
        weights = weights / s if s > 0 else weights

    ess = float(1.0 / np.sum(weights ** 2)) if weights.size else 0.0

    # Parameters from posterior_samples.npz if available
    params_summary: Dict[str, Dict[str, float]] = {}
    if os.path.isfile(post_path):
        try:
            data = np.load(post_path)
            # Expect arrays: names (object array of str) and samples (N x D), and optionally weights
            names = None
            if "names" in data:
                names = [str(n) for n in data["names"]]
            samples = None
            for key in ("samples", "posterior_samples", "xs"):
                if key in data:
                    samples = np.asarray(data[key])
                    break
            w_post = None
            for key in ("weights", "w"):
                if key in data:
                    w_post = np.asarray(data[key], dtype=float)
                    break
            if w_post is None:
                w_post = weights if weights.size else None

            if samples is not None:
                N, D = samples.shape[0], samples.shape[1] if samples.ndim == 2 else (samples.shape[0], 1)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 1)
                if names is None or len(names) != samples.shape[1]:
                    names = [f"param_{i}" for i in range(samples.shape[1])]
                if w_post is None or w_post.size != samples.shape[0]:
                    # fallback to uniform
                    w_post = np.ones(samples.shape[0], dtype=float) / float(samples.shape[0])
                else:
                    s = np.sum(w_post)
                    if s > 0:
                        w_post = w_post / s
                    else:
                        w_post = np.ones(samples.shape[0], dtype=float) / float(samples.shape[0])

                for j, name in enumerate(names):
                    x = samples[:, j]
                    mean, std = _weighted_stats_1d(x, w_post)
                    q16, q50, q84 = _quantiles_1d(x, w_post, qs=(0.16, 0.5, 0.84))
                    params_summary[name] = {
                        "mean": float(mean),
                        "std": float(std),
                        "p16": float(q16),
                        "p50": float(q50),
                        "p84": float(q84),
                    }

                # Plots: 1D marginals and pairwise scatter for top 4 params
                outdir = os.path.join(run_dir, "post_analysis")
                os.makedirs(outdir, exist_ok=True)

                # Weight histogram
                fig, ax = plt.subplots(figsize=(6, 4))
                w_for_plot = w_post if w_post is not None else weights
                ax.hist(w_for_plot, bins=50, color="#4C78A8")
                ax.set_xlabel("Posterior weight")
                ax.set_ylabel("Count")
                ax.set_title("Posterior weights histogram")
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, "weights_hist.png"), dpi=150)
                plt.close(fig)

                # logZ trace if available
                if logz.size > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(np.arange(len(logz)), logz, lw=1.5)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("logZ")
                    ax.set_title("Evidence (logZ) trace")
                    fig.tight_layout()
                    fig.savefig(os.path.join(outdir, "logz_trace.png"), dpi=150)
                    plt.close(fig)

                # 1D marginals for up to 8 parameters
                max_1d = min(samples.shape[1], 8)
                cols = 4
                rows = int(math.ceil(max_1d / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
                axes = np.atleast_1d(axes).ravel()
                for j in range(max_1d):
                    ax = axes[j]
                    x = samples[:, j]
                    ax.hist(x, bins=60, weights=w_post, color="#72B7B2")
                    ax.set_title(names[j])
                for k in range(max_1d, len(axes)):
                    axes[k].axis("off")
                fig.suptitle("1D posterior marginals (weighted)")
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                fig.savefig(os.path.join(outdir, "marginals_1d.png"), dpi=150)
                plt.close(fig)

                # Simple pairwise scatter for top 4 parameters
                top = min(samples.shape[1], 4)
                if top >= 2:
                    fig, axes = plt.subplots(top, top, figsize=(3 * top, 3 * top))
                    for i in range(top):
                        for j in range(top):
                            ax = axes[i, j]
                            if i == j:
                                ax.hist(samples[:, j], bins=50, weights=w_post, color="#E45756")
                            else:
                                ax.scatter(samples[:, j], samples[:, i], s=np.clip(50.0 * w_post, 1.0, 5.0), alpha=0.3)
                            if i == top - 1:
                                ax.set_xlabel(names[j])
                            if j == 0:
                                ax.set_ylabel(names[i])
                    fig.suptitle("Pairwise projections (top parameters)")
                    fig.tight_layout(rect=[0, 0, 1, 0.97])
                    fig.savefig(os.path.join(outdir, "pairs_top.png"), dpi=150)
                    plt.close(fig)
        except Exception as e:
            # If posterior_samples.npz missing or malformed, continue with minimal report
            params_summary["__error_loading_posterior_samples__"] = {"message": str(e)}

    # Posterior summary object
    n_samples = int(weights.size) if isinstance(weights, np.ndarray) else 0
    ps = PosteriorSummary(
        n_samples=n_samples,
        ess=float(ess),
        ess_ratio=(float(ess) / float(n_samples) if n_samples > 0 else 0.0),
        has_nan_weights=has_nan,
        has_neg_weights=has_neg,
        weight_min=(float(np.min(weights)) if weights.size else 0.0),
        weight_max=(float(np.max(weights)) if weights.size else 0.0),
    )

    summary = RunSummary(
        run_dir=run_dir,
        created_at=datetime.utcnow().isoformat(),
        evidence=ev,
        posterior=ps,
        params=params_summary,
        notes=(
            "Weights derived from logwt where available; ESS uses normalized posterior weights. "
            "Evidence summary reflects the final logZ/logZerr trace entry from results.pkl."
        ),
    )

    # Save JSON
    outdir = os.path.join(run_dir, "post_analysis")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "post_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": summary.run_dir,
                "created_at": summary.created_at,
                "evidence": asdict(summary.evidence),
                "posterior": asdict(summary.posterior),
                "params": summary.params,
                "notes": summary.notes,
            },
            f,
            indent=2,
        )

    # Save parameter CSV and LaTeX tables
    # CSV
    import csv
    params_csv = os.path.join(outdir, "params_summary.csv")
    with open(params_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["parameter", "mean", "std", "p16", "p50", "p84"])
        for k, p in summary.params.items():
            if k.startswith("__error"):
                continue
            writer.writerow([k, p.get("mean"), p.get("std"), p.get("p16"), p.get("p50"), p.get("p84")])

    # LaTeX table for parameters
    params_tex = os.path.join(outdir, "params_table.tex")
    with open(params_tex, "w", encoding="utf-8") as tf:
        tf.write("% Auto-generated parameter summary table\n")
        tf.write("\\begin{table}[htbp]\\centering\n")
        tf.write("\\caption{Posterior parameter summary (weighted).}\\label{tab:params}\n")
        tf.write("\\begin{tabular}{lrrrrr}\\hline\n")
        tf.write("Parameter & Mean & Std & 16th & 50th & 84th \\\\ \\hline\n")
        for k, p in summary.params.items():
            if k.startswith("__error"):
                continue
            tf.write(f"{k} & {p.get('mean')} & {p.get('std')} & {p.get('p16')} & {p.get('p50')} & {p.get('p84')} \\\\ \n")
        tf.write("\\hline\\end{tabular}\n\\end{table}\n")

    # LaTeX table for evidence
    evidence_tex = os.path.join(outdir, "evidence_table.tex")
    with open(evidence_tex, "w", encoding="utf-8") as ef:
        ef.write("% Auto-generated evidence summary table\n")
        ef.write("\\begin{table}[htbp]\\centering\n")
        ef.write("\\caption{Nested sampling evidence summary.}\\label{tab:evidence}\n")
        ef.write("\\begin{tabular}{lrr}\\hline\n")
        ef.write("Quantity & Value & Notes \\\\ \\hline\n")
        ef.write(f"Final $\\log Z$ & {summary.evidence.logz_final} & Final entry of trace \\\\ \n")
        ef.write(f"Final $\\sigma(\\log Z)$ & {summary.evidence.logzerr_final} & Reported by dynesty \\\\ \n")
        ef.write(f"Trace length & {summary.evidence.n_logz_points} & iterations \\\\ \n")
        ef.write(f"ESS & {summary.posterior.ess:.3f} & normalized posterior weights \\\\ \n")
        ef.write(f"ESS ratio & {summary.posterior.ess_ratio:.6f} & ESS / N \\\\ \n")
        ef.write("\\hline\\end{tabular}\n\\end{table}\n")

    # Save Markdown report
    md_path = os.path.join(outdir, "post_analysis_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Post-analysis Report\n\n")
        f.write(f"Run directory: `{run_dir}`\n\n")
        f.write("## Evidence\n\n")
        f.write(f"- Final logZ: {summary.evidence.logz_final}\n")
        f.write(f"- Final logZerr: {summary.evidence.logzerr_final}\n")
        f.write(f"- Points in logZ trace: {summary.evidence.n_logz_points}\n\n")
        f.write("## Posterior weights\n\n")
        f.write(f"- Samples with weights: {summary.posterior.n_samples}\n")
        f.write(f"- ESS: {summary.posterior.ess:.3f}\n")
        f.write(f"- ESS ratio: {summary.posterior.ess_ratio:.6f}\n")
        f.write(f"- Any NaN weights: {summary.posterior.has_nan_weights}\n")
        f.write(f"- Any negative weights: {summary.posterior.has_neg_weights}\n")
        f.write(f"- Weight min/max: {summary.posterior.weight_min:.3e} / {summary.posterior.weight_max:.3e}\n\n")
        f.write("![Weights histogram](weights_hist.png)\n\n")
        if ev.n_logz_points > 0:
            f.write("![logZ trace](logz_trace.png)\n\n")
        if params_summary:
            f.write("## Parameter summaries (selected)\n\n")
            # Print first up to 10 parameters
            keys = [k for k in params_summary.keys() if not k.startswith("__error")] \
                   or list(params_summary.keys())
            for k in keys[:10]:
                p = params_summary[k]
                f.write(f"- {k}: mean={p.get('mean')}, std={p.get('std')}, "
                        f"p16={p.get('p16')}, p50={p.get('p50')}, p84={p.get('p84')}\n")
            f.write("\n![1D marginals](marginals_1d.png)\n\n")
            if os.path.isfile(os.path.join(outdir, "pairs_top.png")):
                f.write("![Pairwise projections](pairs_top.png)\n\n")
        if any(k.startswith("__error") for k in params_summary.keys()):
            f.write("\nNote: Errors occurred while reading posterior_samples.npz. See JSON for details.\n")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a single run directory")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary = analyze_run(run_dir)
    print(json.dumps({
        "run_dir": summary.run_dir,
        "evidence": asdict(summary.evidence),
        "posterior": asdict(summary.posterior),
        "n_params": len(summary.params),
    }, indent=2))


if __name__ == "__main__":
    main()
