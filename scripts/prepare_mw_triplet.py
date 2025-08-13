#!/usr/bin/env python3
import argparse
import json
import pathlib
from datetime import datetime
from typing import Optional, Dict, Any

DOCS_MD_PATH = pathlib.Path("docs/mw_triplet.md")
DOCS_JSON_PATH = pathlib.Path("docs/mw_triplet_summary.json")
IMAGES_DIR = pathlib.Path("images")


def load_run(run_dir: pathlib.Path) -> Dict[str, Any]:
    enh = run_dir / "run_summary_enhanced.json"
    basic = run_dir / "run_summary.json"
    if enh.exists():
        with enh.open("r", encoding="utf-8") as f:
            return json.load(f)
    with basic.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_logz(summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    ev = summary.get("evidence_metrics", {})
    logz = ev.get("logz")
    logz_err = ev.get("logz_error")
    if logz is None and "evidence" in summary:
        # legacy structure fallback
        logz = summary["evidence"].get("logz")
        logz_err = summary["evidence"].get("logz_error")
    if logz is None:
        return None
    return {"logz": float(logz), "logz_error": float(logz_err) if logz_err is not None else None}


def render_md(data: Dict[str, Any]) -> str:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    lines = []
    lines.append("# Milky Way Evidence Triplet (GR, TFR/tidal, NFW)")
    lines.append("")
    lines.append(f"Last updated: {ts}")
    lines.append("")
    lines.append("This document aggregates the Bayesian evidence and summary context for three Milky Way runs under matched settings: GR (baryons only), TFR/tidal (environmental coupling), and NFW (ΛCDM halo). The tidal entry is left as a placeholder until its run completes.")
    lines.append("")
    # Evidence table
    lines.append("Evidence (log Z ± σ):")
    lines.append("")
    def fmt(entry: Optional[Dict[str, float]]):
        if not entry:
            return "—"
        z = entry["logz"]
        e = entry.get("logz_error")
        return f"{z:.2f} ± {e:.2f}" if e is not None else f"{z:.2f}"
    gr = data.get("gr")
    tfr = data.get("tfr")
    nfw = data.get("nfw")
    lines.append(f"- GR:  {fmt(gr)}")
    lines.append(f"- TFR: {fmt(tfr)}")
    lines.append(f"- NFW: {fmt(nfw)}")
    lines.append("")
    # Pairwise deltas where possible
    if gr and nfw:
        dnfw_gr = nfw["logz"] - gr["logz"]
        lines.append(f"Delta log Z (NFW − GR): {dnfw_gr:.2f}")
    else:
        lines.append("Delta log Z (NFW − GR): —")
    if gr and tfr:
        dtfr_gr = tfr["logz"] - gr["logz"]
        lines.append(f"Delta log Z (TFR − GR): {dtfr_gr:.2f}")
    else:
        lines.append("Delta log Z (TFR − GR): — (pending tidal)")
    if tfr and nfw:
        dnfw_tfr = nfw["logz"] - tfr["logz"]
        lines.append(f"Delta log Z (NFW − TFR): {dnfw_tfr:.2f}")
    else:
        lines.append("Delta log Z (NFW − TFR): — (pending tidal)")
    lines.append("")
    lines.append("Notes:")
    lines.append("- GR and NFW values are sourced from the run_summary_enhanced.json files in their respective run folders under runs/.")
    lines.append("- The TFR/tidal entry will be filled automatically when its run_summary is available; re-run the prep script to update.")
    lines.append("")
    # Image placeholder guidance
    lines.append("Figures:")
    lines.append("- images/mw_rotation_triplet.png: reserved composite overlay (auto-regenerated when tidal is available).")
    lines.append("- images/mw_rotation_gr.png and images/mw_rotation_nfw.png: single-model overlays if generated.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Prepare MW triplet doc and summary from run dirs")
    ap.add_argument("--gr", type=pathlib.Path, required=True, help="Path to GR run dir")
    ap.add_argument("--nfw", type=pathlib.Path, required=True, help="Path to NFW run dir")
    ap.add_argument("--tfr", type=pathlib.Path, default=None, help="Path to TFR/tidal run dir (optional)")
    ap.add_argument("--write-md", action="store_true", help="Write/update docs/mw_triplet.md")
    ap.add_argument("--write-json", action="store_true", help="Write/update docs/mw_triplet_summary.json")
    args = ap.parse_args()

    out: Dict[str, Any] = {}

    # GR
    gr_sum = load_run(args.gr)
    gr_ev = extract_logz(gr_sum)
    if gr_ev:
        out["gr"] = gr_ev
        out["gr"]["run_dir"] = str(args.gr).replace("\\", "/")
    # NFW
    nfw_sum = load_run(args.nfw)
    nfw_ev = extract_logz(nfw_sum)
    if nfw_ev:
        out["nfw"] = nfw_ev
        out["nfw"]["run_dir"] = str(args.nfw).replace("\\", "/")
    # TFR optional
    if args.tfr:
        tfr_dir = pathlib.Path(args.tfr)
        if (tfr_dir / "run_summary_enhanced.json").exists() or (tfr_dir / "run_summary.json").exists():
            tfr_sum = load_run(tfr_dir)
            tfr_ev = extract_logz(tfr_sum)
            if tfr_ev:
                out["tfr"] = tfr_ev
                out["tfr"]["run_dir"] = str(tfr_dir).replace("\\", "/")

    # Write JSON summary
    if args.write_json:
        DOCS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DOCS_JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # Write markdown doc
    if args.write_md:
        md = render_md(out)
        DOCS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DOCS_MD_PATH.open("w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()

