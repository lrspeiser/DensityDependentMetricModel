#!/usr/bin/env python3
"""
Aggregate per-galaxy SPARC evidence JSON outputs into a CSV and Markdown table.

- Scans for JSON files under images/ and data/ matching sparc_*.json
- Expects each JSON to include at minimum:
  {
    "galaxy": "NGC3198",
    "model": "ER" | "GR" | "NFW",
    "logZ": float,
    "logZ_err": float or null,
    "sigma_floor": float or null,
    "t_proxy": str or null
  }
- Produces:
  - data/sparc_batch_summary.csv
  - docs/ED-SPARC.md (markdown table snippet)

This is non-interactive and safe to run repeatedly.
"""
import json
import csv
import re
from pathlib import Path
from collections import defaultdict

RE_GAL = re.compile(r"sparc_([a-z0-9]+)\.json", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
DATA = ROOT / "data"
DOCS = ROOT / "docs"
OUTPUT_CSV = DATA / "sparc_batch_summary.csv"
OUTPUT_MD = DOCS / "ED-SPARC.md"

# Ensure directories exist
DATA.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def find_jsons() -> list[Path]:
    candidates = []
    if IMAGES.exists():
        candidates += list(IMAGES.glob("sparc_*.json"))
    if DATA.exists():
        candidates += list(DATA.glob("sparc_*.json"))
    # Deduplicate by path
    seen = set()
    uniq = []
    for p in candidates:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq


def load_entry(p: Path) -> dict | None:
    try:
        with p.open("r", encoding="utf-8") as f:
            j = json.load(f)
        # Normalize keys
        galaxy = j.get("galaxy") or j.get("name") or j.get("target")
        model = (j.get("model") or j.get("mode") or j.get("fit_model") or "").upper()
        logZ = j.get("logZ")
        logZ_err = j.get("logZ_err") or j.get("logZ_unc") or j.get("logZ_sigma")
        sigma_floor = j.get("sigma_floor")
        t_proxy = j.get("t_proxy") or j.get("tidal_proxy") or j.get("T_proxy")
        # Fallback galaxy name from filename
        if not galaxy:
            m = RE_GAL.search(p.name)
            if m:
                galaxy = m.group(1)
        if not (galaxy and model and (logZ is not None)):
            return None
        if model not in {"ER", "GR", "NFW"}:
            # Try normalize common variants
            m2 = model.replace("-", "").replace("_", "")
            if m2 in {"ERENV", "ER"}:
                model = "ER"
            elif m2 in {"GRBARYONS", "GR"}:
                model = "GR"
            elif m2 in {"LCDM", "NFW"}:
                model = "NFW"
            else:
                return None
        return {
            "galaxy": str(galaxy).upper(),
            "model": model,
            "logZ": float(logZ),
            "logZ_err": float(logZ_err) if logZ_err is not None else None,
            "sigma_floor": float(sigma_floor) if sigma_floor is not None else None,
            "t_proxy": t_proxy,
            "source": str(p.relative_to(ROOT)),
        }
    except Exception:
        return None


def aggregate(entries: list[dict]) -> tuple[list[dict], dict]:
    # Group by galaxy then model
    by_gal = defaultdict(dict)
    meta = {}
    for e in entries:
        g = e["galaxy"]
        m = e["model"]
        by_gal[g][m] = e
        if g not in meta:
            meta[g] = {
                "sigma_floor": e.get("sigma_floor"),
                "t_proxy": e.get("t_proxy"),
            }
    rows = []
    for g, md in sorted(by_gal.items()):
        er = md.get("ER")
        gr = md.get("GR")
        nfw = md.get("NFW")
        def val(e, k):
            return e.get(k) if e else None
        row = {
            "galaxy": g,
            "logZ_ER": val(er, "logZ"),
            "logZ_ER_err": val(er, "logZ_err"),
            "logZ_NFW": val(nfw, "logZ"),
            "logZ_NFW_err": val(nfw, "logZ_err"),
            "logZ_GR": val(gr, "logZ"),
            "logZ_GR_err": val(gr, "logZ_err"),
        }
        # Deltas if available
        if row["logZ_ER"] is not None and row["logZ_GR"] is not None:
            row["dlogZ_ER_minus_GR"] = row["logZ_ER"] - row["logZ_GR"]
        else:
            row["dlogZ_ER_minus_GR"] = None
        if row["logZ_NFW"] is not None and row["logZ_GR"] is not None:
            row["dlogZ_NFW_minus_GR"] = row["logZ_NFW"] - row["logZ_GR"]
        else:
            row["dlogZ_NFW_minus_GR"] = None
        if row["logZ_ER"] is not None and row["logZ_NFW"] is not None:
            row["dlogZ_ER_minus_NFW"] = row["logZ_ER"] - row["logZ_NFW"]
        else:
            row["dlogZ_ER_minus_NFW"] = None
        # Carry meta
        row["sigma_floor"] = meta[g]["sigma_floor"]
        row["t_proxy"] = meta[g]["t_proxy"]
        rows.append(row)
    return rows, meta


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        # Still create a header-only CSV for reproducibility
        headers = [
            "galaxy","logZ_ER","logZ_ER_err","logZ_NFW","logZ_NFW_err","logZ_GR","logZ_GR_err",
            "dlogZ_ER_minus_GR","dlogZ_NFW_minus_GR","dlogZ_ER_minus_NFW","sigma_floor","t_proxy",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = []
    lines.append("# Extended Data: SPARC Evidence Summary (Auto-generated)\n")
    lines.append("This file is generated by tools/generate_sparc_ed_table.py. Do not edit manually.\n")
    lines.append("")
    # Table header
    headers = [
        "Galaxy","logZ(ER)","logZ(NFW)","logZ(GR)",
        "ΔlogZ ER−GR","ΔlogZ NFW−GR","ΔlogZ ER−NFW","σ_floor [km s⁻¹]","T proxy",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([":-" for _ in headers]) + "|")
    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)
    for r in rows:
        line = [
            r.get("galaxy"),
            fmt(r.get("logZ_ER")),
            fmt(r.get("logZ_NFW")),
            fmt(r.get("logZ_GR")),
            fmt(r.get("dlogZ_ER_minus_GR")),
            fmt(r.get("dlogZ_NFW_minus_GR")),
            fmt(r.get("dlogZ_ER_minus_NFW")),
            fmt(r.get("sigma_floor")),
            r.get("t_proxy") or "",
        ]
        lines.append("| " + " | ".join(line) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    paths = find_jsons()
    entries = []
    for p in paths:
        e = load_entry(p)
        if e:
            entries.append(e)
    rows, _meta = aggregate(entries)
    write_csv(rows, OUTPUT_CSV)
    write_markdown(rows, OUTPUT_MD)
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_MD} with {len(rows)} galaxies.")


if __name__ == "__main__":
    main()
