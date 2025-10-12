#!/usr/bin/env python3
"""
import_slacs_asu_tsv.py

Parse VizieR ASU-TSV for SLACS (Auger+ 2009; J/ApJ/705/1099/lenses) and emit:

1) A curated, source-like CSV (docs/lensing_targets_slacs_sl2s.csv schema)
   columns: lens_name,survey,z_l,z_s,theta_E_arcsec,theta_E_err_arcsec,
            log10Mstar_chab,log10Mstar_chab_err,Re_arcsec,n_sersic,
            q_axis_ratio,band_for_Re,notes

2) An orchestrator-ready CSV (docs/lensing_targets_slacs.csv) matching
   scripts/next_steps_from_run.py expectations:
   columns: lens_id,z_l,z_s,log10M_star,Re_kpc,n_sersic,
            theta_E_obs_arcsec,theta_E_obs_err_arcsec,profile,notes

Notes
- The ASU-TSV includes column definitions as #Column lines; the data lines
  have no header. This script reconstructs the header order from the metadata.
- We choose Re(I) when present, else Re(V), else Re(B); we record band_for_Re.
- We DO NOT synthesize theta_E errors; theta_E_err_arcsec is left blank.
- n_sersic is not provided by this table; we default to 4 (ETG baseline),
  matching repository conventions for missing n.
- Cosmology for Re_kpc conversion: flat LCDM, H0=70 km/s/Mpc, Om=0.3.
  This conversion is used only for the orchestrator-ready CSV.

Citations
- Auger et al. 2009, ApJ 705, 1099 (SLACS IX). VizieR: J/ApJ/705/1099

"""
import argparse
import math
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Import SLACS ASU-TSV and emit curated/orchestrator CSVs")
    ap.add_argument("--in", dest="in_path", default="data/asu.tsv", help="Input ASU-TSV path (VizieR export)")
    ap.add_argument("--out-src", dest="out_src", default="docs/lensing_targets_slacs_sl2s.csv", help="Output curated CSV (extended schema)")
    ap.add_argument("--out-orch", dest="out_orch", default="docs/lensing_targets_slacs.csv", help="Output orchestrator-ready CSV")
    return ap.parse_args()


def read_asu_tsv(path: Path):
    cols = []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#Column"):
                # Format: #Column\tNAME\t(TYPE)\t...; we want NAME, but beware of names like 'Re(I)'
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    name = parts[1].strip()
                    cols.append(name)
                continue
            if line.startswith("#"):
                continue
            # Data line
            vals = line.rstrip("\n").split("\t")
            # Skip in-band header/unit/separator rows sometimes emitted by VizieR ASU-TSV
            if vals and vals[0].strip() == "SDSS":
                # header row (column names echoed)
                continue
            if vals and vals[0].startswith("----------"):
                # separator row
                continue
            if any(tok.strip() in ("km/s", "mag", "arcsec", "[Msun]", '"h:m:s"', '"d:m:s"') for tok in vals):
                # units row
                continue
            # If there are fewer values than columns, pad with blanks
            if len(vals) < len(cols):
                vals += [""] * (len(cols) - len(vals))
            row = dict(zip(cols, vals))
            rows.append(row)
    return cols, rows


def safe_float(s):
    try:
        s = (s or "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def kpc_per_arcsec_flatlcdm(z: float, H0: float = 70.0, Om: float = 0.3) -> float:
    """Angular scale [kpc/arcsec] in flat LCDM with given H0, Om.
    Uses a simple numerical integral for comoving distance.
    """
    if z is None or z <= 0:
        return None
    c = 299792.458  # km/s
    # Integrate 1/E(z) dz via Simpson-like composite trapezoid
    n = 2048
    dz = z / n
    acc = 0.0
    for i in range(n + 1):
        zi = i * dz
        Ez = math.sqrt(Om * (1.0 + zi) ** 3 + (1.0 - Om))
        w = 4.0 if i % 2 == 1 else 2.0
        if i == 0 or i == n:
            w = 1.0
        acc += w / Ez
    I = acc * dz / 3.0
    DM_Mpc = (c / H0) * I
    DA_Mpc = DM_Mpc / (1.0 + z)
    kpc_per_arcsec = DA_Mpc * 1e3 * (math.pi / (180.0 * 3600.0))
    return kpc_per_arcsec


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_src = Path(args.out_src)
    out_orch = Path(args.out_orch)
    out_src.parent.mkdir(parents=True, exist_ok=True)
    out_orch.parent.mkdir(parents=True, exist_ok=True)

    cols, rows = read_asu_tsv(in_path)
    # Build outputs
    src_lines = [
        "lens_name,survey,z_l,z_s,theta_E_arcsec,theta_E_err_arcsec,log10Mstar_chab,log10Mstar_chab_err,Re_arcsec,n_sersic,q_axis_ratio,band_for_Re,notes"
    ]
    orch_lines = [
        "lens_id,z_l,z_s,log10M_star,Re_kpc,n_sersic,theta_E_obs_arcsec,theta_E_obs_err_arcsec,profile,notes,q_axis_ratio"
    ]

    for r in rows:
        name = (r.get("SDSS") or "").strip()
        z_l = safe_float(r.get("zlens"))
        z_s = safe_float(r.get("zsrc"))
        RE_kpc = safe_float(r.get("RE"))  # SLACS table provides Einstein radius in kpc
        logMc = safe_float(r.get("logMc"))
        elogMc = safe_float(r.get("e_logMc"))
        # Choose Re in band priority I, V, B
        ReI = safe_float(r.get("Re(I)"))
        ReV = safe_float(r.get("Re(V)"))
        ReB = safe_float(r.get("Re(B)"))
        Re_arcsec = None
        band = ""
        for band_name, val in (("I", ReI), ("V", ReV), ("B", ReB)):
            if val is not None:
                Re_arcsec = val
                band = band_name
                break
        # Compose notes
        mtype = (r.get("MType") or "").strip()
        comp = (r.get("f_MType") or "").strip()
        notes = f"SLACS Auger+2009; MType={mtype}{'; companion' if comp=='c' else ''}; Re_band={band}"
        # Try to recover axis ratio q from common variants if present
        q_candidates = []
        for key in r.keys():
            lk = key.lower()
            if lk in ("q", "axisratio", "b/a") or lk.startswith("q(") or "axis" in lk or "b/a" in lk or "ellip" in lk:
                q_candidates.append(key)
        q_axis = None
        for k in q_candidates:
            try:
                v = safe_float(r.get(k))
                if v is not None and v > 0 and v <= 1.0:
                    q_axis = v
                    break
            except Exception:
                pass

        # 1) Source-like CSV row
        # Convert RE_kpc to arcsec if possible
        th_arc = None
        if (RE_kpc is not None) and (z_l is not None):
            kpc_per_arc = kpc_per_arcsec_flatlcdm(z_l)
            if kpc_per_arc is not None and kpc_per_arc > 0:
                th_arc = RE_kpc / kpc_per_arc
        src_row = [
            name,
            "SLACS",
            f"{z_l if z_l is not None else ''}",
            f"{z_s if z_s is not None else ''}",
            f"{th_arc if th_arc is not None else ''}",
            "",  # theta_E_err_arcsec unknown in table
            f"{logMc if logMc is not None else ''}",
            f"{elogMc if elogMc is not None else ''}",
            f"{Re_arcsec if Re_arcsec is not None else ''}",
            "4",  # n_sersic default (ETG)
            f"{q_axis if q_axis is not None else ''}",
            band,
            notes + "; RE_kpc from SLACS converted to arcsec",
        ]
        src_lines.append(",".join(src_row))

        # 2) Orchestrator-ready row (requires Re_kpc)
        Re_kpc = None
        if (Re_arcsec is not None) and (z_l is not None):
            kpc_per_arc = kpc_per_arcsec_flatlcdm(z_l)
            if kpc_per_arc is not None:
                Re_kpc = Re_arcsec * kpc_per_arc
        orch_row = [
            name,
            f"{z_l if z_l is not None else ''}",
            f"{z_s if z_s is not None else ''}",
            f"{logMc if logMc is not None else ''}",
            f"{Re_kpc if Re_kpc is not None else ''}",
            "4",  # n_sersic default
            f"{th_arc if th_arc is not None else ''}",
            "",  # theta_E_obs_err_arcsec unknown
            "sersic",
            notes + "; RE_kpc from SLACS converted to arcsec",
            f"{q_axis if q_axis is not None else ''}",
        ]
        orch_lines.append(",".join(orch_row))

    out_src.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    out_orch.write_text("\n".join(orch_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(src_lines)-1} rows -> {out_src}")
    print(f"Wrote {len(orch_lines)-1} rows -> {out_orch}")


if __name__ == "__main__":
    main()
