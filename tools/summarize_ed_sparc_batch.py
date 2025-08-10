#!/usr/bin/env python3
"""
Summarize ED-SPARC batch evidence results.
- Reads ed_sparc_batch.csv (galaxy, logZ_GR, logZ_NFW, logZ_ER, ...)
- Computes Bayes factors (ΔlogZ) and simple ranks
- Writes:
  - docs/ed_sparc_summary.md (human-readable summary)
  - docs/ed_sparc_table.md (markdown table)
- Prints a short CLI summary to stdout
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict, Any

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / 'ed_sparc_batch.csv'
DOC_SUMMARY = REPO / 'docs' / 'ed_sparc_summary.md'
DOC_TABLE = REPO / 'docs' / 'ed_sparc_table.md'


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def compute_metrics(rows: List[Dict[str, Any]]) -e List[Dict[str, Any]]:
    out = []
    for row in rows:
        z_gr = safe_float(row.get('logZ_GR'))
        z_nfw = safe_float(row.get('logZ_NFW'))
        z_er = safe_float(row.get('logZ_ER'))
        z_comp = safe_float(row.get('logZ_COMP'))
        d_er_gr = (z_er - z_gr) if (z_er is not None and z_gr is not None) else None
        d_er_nfw = (z_er - z_nfw) if (z_er is not None and z_nfw is not None) else None
        d_comp_gr = (z_comp - z_gr) if (z_comp is not None and z_gr is not None) else None
        d_comp_nfw = (z_comp - z_nfw) if (z_comp is not None and z_nfw is not None) else None
        d_comp_er = (z_comp - z_er) if (z_comp is not None and z_er is not None) else None
        out.append({
            'galaxy': row.get('galaxy'),
            'logZ_GR': z_gr,
            'logZ_NFW': z_nfw,
            'logZ_ER': z_er,
            'logZ_COMP': z_comp,
            'dER_GR': d_er_gr,
            'dER_NFW': d_er_nfw,
            'dCOMP_GR': d_comp_gr,
            'dCOMP_NFW': d_comp_nfw,
            'dCOMP_ER': d_comp_er,
        })
    return out
    return out


def format_float(x, nd=2):
    if x is None:
        return '—'
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return '—'


def write_markdown_table(items: List[Dict[str, Any]]):
    DOC_TABLE.parent.mkdir(parents=True, exist_ok=True)
    headers = ['Galaxy','logZ_GR','logZ_NFW','logZ_ER','logZ_COMP','ΔlogZ(ER−GR)','ΔlogZ(ER−NFW)','ΔlogZ(COMP−GR)','ΔlogZ(COMP−NFW)','ΔlogZ(COMP−ER)']
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    for r in items:
        lines.append("| " + " | ".join([
            str(r['galaxy']),
            format_float(r['logZ_GR']),
            format_float(r['logZ_NFW']),
            format_float(r['logZ_ER']),
            format_float(r['logZ_COMP']),
            format_float(r['dER_GR']),
            format_float(r['dER_NFW']),
            format_float(r['dCOMP_GR']),
            format_float(r['dCOMP_NFW']),
            format_float(r['dCOMP_ER']),
        ]) + " |")
    DOC_TABLE.write_text("\n".join(lines), encoding='utf-8')


def write_summary(items: List[Dict[str, Any]]):
    # Basic counts
    n = len(items)
    n_pref_er_over_gr = sum(1 for r in items if r['dER_GR'] is not None and r['dER_GR'] > 0)
    n_pref_er_over_nfw = sum(1 for r in items if r['dER_NFW'] is not None and r['dER_NFW'] > 0)
    n_pref_comp_over_gr = sum(1 for r in items if r['dCOMP_GR'] is not None and r['dCOMP_GR'] > 0)
    n_pref_comp_over_er = sum(1 for r in items if r['dCOMP_ER'] is not None and r['dCOMP_ER'] > 0)

    # Top/bottom by ER−GR
    by_er_gr = [r for r in items if r['dER_GR'] is not None]
    by_er_gr.sort(key=lambda r: r['dER_GR'], reverse=True)
    top5 = by_er_gr[:5]
    bottom5 = by_er_gr[-5:]

    lines = []
    lines.append("# ED-SPARC Evidence Summary")
    lines.append("")
    lines.append(f"- Galaxies analyzed: {n}")
    lines.append(f"- ER/TFR preferred over GR (ΔlogZ>0): {n_pref_er_over_gr}/{n}")
    lines.append(f"- ER/TFR preferred over NFW (ΔlogZ>0): {n_pref_er_over_nfw}/{n}")
    lines.append(f"- Composite (ER+NFW) preferred over GR (ΔlogZ>0): {n_pref_comp_over_gr}/{n}")
    lines.append(f"- Composite preferred over ER/TFR (ΔlogZ>0): {n_pref_comp_over_er}/{n}")
    lines.append("")
    lines.append("## Top 5 by ΔlogZ(ER−GR)")
    for r in top5:
        lines.append(f"- {r['galaxy']}: ΔlogZ(ER−GR)={format_float(r['dER_GR'])}, ΔlogZ(ER−NFW)={format_float(r['dER_NFW'])}")
    lines.append("")
    lines.append("## Bottom 5 by ΔlogZ(ER−GR)")
    for r in bottom5:
        lines.append(f"- {r['galaxy']}: ΔlogZ(ER−GR)={format_float(r['dER_GR'])}, ΔlogZ(ER−NFW)={format_float(r['dER_NFW'])}")
    lines.append("")
    lines.append("See full table in docs/ed_sparc_table.md.")

    DOC_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DOC_SUMMARY.write_text("\n".join(lines), encoding='utf-8')


def main():
    rows = load_rows()
    items = compute_metrics(rows)
    write_markdown_table(items)
    write_summary(items)

    # Print concise CLI summary
    print(f"Wrote {DOC_TABLE} and {DOC_SUMMARY}")
    print("Sample rows (first 5):")
    for r in items[:5]:
        print(f"  {r['galaxy']}: ΔlogZ(ER−GR)={format_float(r['dER_GR'])}, ΔlogZ(ER−NFW)={format_float(r['dER_NFW'])}, ΔlogZ(COMP−ER)={format_float(r['dCOMP_ER'])}")


if __name__ == '__main__':
    main()
