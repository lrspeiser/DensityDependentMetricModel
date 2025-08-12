#!/usr/bin/env python3
"""
Generate Milky Way evidence triplet summary and ΔlogZ plot.

Inputs: three run directories (GR, TFR/tidal_band, NFW) that contain
- run_summary_enhanced.json (preferred) or run_summary.json
- posterior_samples.npz (optional; not required here)

Outputs:
- docs/mw_triplet.md (markdown summary with logZ and deltas)
- images/mw_dlogz_triplet.png (bar plot of ΔlogZ vs GR)
- results/mw_triplet_summary.json (machine-readable record)

Usage:
  python scripts/generate_mw_triplet.py \
    --gr runs/gr_20250811_232403 \
    --tfr runs/tidal_band_20250810_102330 \
    --nfw runs/nfw_20250812_082825
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    candidates = [
        run_dir / "run_summary_enhanced.json",
        run_dir / "run_summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    raise FileNotFoundError(f"No summary JSON found in {run_dir}")


def _extract_logz(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # Prefer evidence_metrics if present
    ev = summary.get("evidence_metrics", {})
    logz = ev.get("logz")
    logz_err = ev.get("logz_error")
    if logz is not None:
        try:
            return float(logz), (None if logz_err is None else float(logz_err))
        except Exception:
            pass
    # Fallback to convergence_metrics.current_logz
    conv = summary.get("convergence_metrics", {})
    cz = conv.get("current_logz")
    if cz is not None:
        try:
            return float(cz), None
        except Exception:
            pass
    # Fallback to top-level fields
    for key in ("logZ", "logz"):
        if key in summary:
            try:
                return float(summary[key]), None
            except Exception:
                pass
    return None, None


def _markdown_table(gr: Tuple[float, Optional[float]],
                    tfr: Tuple[float, Optional[float]],
                    nfw: Tuple[float, Optional[float]]) -> str:
    logZ_GR, eGR = gr
    logZ_TFR, eTFR = tfr
    logZ_NFW, eNFW = nfw

    def fmt(x: Optional[float]) -> str:
        return "—" if x is None else f"{x:.6f}"

    d_TFR_GR = None if (logZ_TFR is None or logZ_GR is None) else (logZ_TFR - logZ_GR)
    d_NFW_GR = None if (logZ_NFW is None or logZ_GR is None) else (logZ_NFW - logZ_GR)
    d_TFR_NFW = None if (logZ_TFR is None or logZ_NFW is None) else (logZ_TFR - logZ_NFW)

    md = []
    md.append("# Milky Way Evidence Triplet (GR vs TFR vs NFW)\n")
    md.append("\n")
    md.append("| Model | logZ | ± err |\n")
    md.append("|-------|------|-------|\n")
    md.append(f"| GR   | {fmt(logZ_GR)} | {fmt(eGR)} |\n")
    md.append(f"| TFR  | {fmt(logZ_TFR)} | {fmt(eTFR)} |\n")
    md.append(f"| NFW  | {fmt(logZ_NFW)} | {fmt(eNFW)} |\n")
    md.append("\n")
    md.append("Deltas (Jeffreys scale reference):\n\n")
    md.append(f"- ΔlogZ(TFR−GR) = {fmt(d_TFR_GR)}\n")
    md.append(f"- ΔlogZ(NFW−GR) = {fmt(d_NFW_GR)}\n")
    md.append(f"- ΔlogZ(TFR−NFW) = {fmt(d_TFR_NFW)}\n")
    md.append("\n")
    return "".join(md)


def _plot_deltas(d_tfr_gr: Optional[float], d_nfw_gr: Optional[float], out_path: Path):
    labels = ["TFR − GR", "NFW − GR"]
    vals = [d_tfr_gr if d_tfr_gr is not None else np.nan,
            d_nfw_gr if d_nfw_gr is not None else np.nan]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=["#D62728", "#2CA02C"])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("ΔlogZ")
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.1f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", required=True, help="Path to GR run directory")
    ap.add_argument("--tfr", required=True, help="Path to TFR (tidal_band) run directory")
    ap.add_argument("--nfw", required=True, help="Path to NFW run directory")
    args = ap.parse_args()

    gr_dir = Path(args.gr)
    tfr_dir = Path(args.tfr)
    nfw_dir = Path(args.nfw)

    gr_sum = _load_summary(gr_dir)
    tfr_sum = _load_summary(tfr_dir)
    nfw_sum = _load_summary(nfw_dir)

    logZ_GR, eGR = _extract_logz(gr_sum)
    logZ_TFR, eTFR = _extract_logz(tfr_sum)
    logZ_NFW, eNFW = _extract_logz(nfw_sum)

    # Prepare outputs
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    images_dir = repo_root / "images"
    results_dir = repo_root / "results"

    # Write markdown
    md = _markdown_table((logZ_GR, eGR), (logZ_TFR, eTFR), (logZ_NFW, eNFW))
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "mw_triplet.md").write_text(md, encoding="utf-8")

    # Plot ΔlogZ bars
    d_TFR_GR = None if (logZ_TFR is None or logZ_GR is None) else (logZ_TFR - logZ_GR)
    d_NFW_GR = None if (logZ_NFW is None or logZ_GR is None) else (logZ_NFW - logZ_GR)
    _plot_deltas(d_TFR_GR, d_NFW_GR, images_dir / "mw_dlogz_triplet.png")

    # Machine-readable summary
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "mw_triplet_summary.json").write_text(json.dumps({
        "runs": {
            "gr": str(gr_dir),
            "tfr": str(tfr_dir),
            "nfw": str(nfw_dir),
        },
        "logZ": {
            "GR": logZ_GR,
            "TFR": logZ_TFR,
            "NFW": logZ_NFW,
        },
        "logZ_err": {
            "GR": eGR,
            "TFR": eTFR,
            "NFW": eNFW,
        },
        "deltas": {
            "TFR_minus_GR": d_TFR_GR,
            "NFW_minus_GR": d_NFW_GR,
            "TFR_minus_NFW": None if (logZ_TFR is None or logZ_NFW is None) else (logZ_TFR - logZ_NFW),
        }
    }, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {docs_dir / 'mw_triplet.md'}")
    print(f"- {images_dir / 'mw_dlogz_triplet.png'}")
    print(f"- {results_dir / 'mw_triplet_summary.json'}")


if __name__ == "__main__":
    main()
