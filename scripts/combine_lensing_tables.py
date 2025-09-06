import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Combine multiple lensing_rar_table.csv files with run labels.")
    p.add_argument("--inputs", nargs="+", required=True, help="List of input CSV files to combine.")
    p.add_argument("--labels", nargs="+", required=True, help="List of labels (same length as inputs).")
    p.add_argument("--out", required=True, help="Output CSV file path (long-format, includes run_label).")
    return p.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def write_csv(path, fieldnames, rows):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pivot_long_to_wide(rows: List[Dict[str, str]], key_field: str, value_field: str, run_label_field: str = "run_label"):
    # Collect unique keys and run labels
    keys = []
    run_labels = []
    for r in rows:
        k = r.get(key_field, "")
        if k not in keys:
            keys.append(k)
        lab = r.get(run_label_field, "")
        if lab not in run_labels:
            run_labels.append(lab)

    # Build mapping key -> {run_label: value}
    mapping = {k: {lab: "" for lab in run_labels} for k in keys}
    for r in rows:
        k = r.get(key_field, "")
        lab = r.get(run_label_field, "")
        mapping.setdefault(k, {})
        if lab in mapping[k]:
            mapping[k][lab] = r.get(value_field, "")
        else:
            mapping[k][lab] = r.get(value_field, "")

    # Create wide rows
    fieldnames = [key_field] + run_labels
    wide_rows = []
    for k in keys:
        row = {key_field: k}
        for lab in run_labels:
            row[lab] = mapping[k].get(lab, "")
        wide_rows.append(row)
    return fieldnames, wide_rows


def main():
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same length")

    combined_rows = []
    base_fields = None

    for label, path in zip(args.labels, args.inputs):
        fields, rows = read_csv(path)
        if base_fields is None:
            base_fields = fields.copy()
            # Add a column for the run label if not present
            if "run_label" not in base_fields:
                base_fields = ["run_label"] + base_fields
        else:
            # Sanity: ensure the same header set (order can differ slightly). If mismatch, union.
            if set(fields) != set(base_fields) - {"run_label"}:
                # Union headers while preserving order as much as possible
                union = [fn for fn in base_fields if fn != "run_label"]
                for fn in fields:
                    if fn not in union:
                        union.append(fn)
                base_fields = ["run_label"] + union
        for r in rows:
            r_out = {k: r.get(k, "") for k in base_fields if k != "run_label"}
            r_out = {"run_label": label, **r_out}
            combined_rows.append(r_out)

    # Write long-format combined CSV (includes run_label)
    write_csv(args.out, base_fields, combined_rows)
    print(f"[OK] Wrote combined lensing table (long): {args.out} ({len(combined_rows)} rows)")

    # Additionally write convenient pivots for the three key prediction columns when present.
    out_dir = os.path.dirname(args.out)
    pivots = [
        ("theta_E_GR_arcsec", os.path.join(out_dir, "lensing_summary_pivot_GR.csv")),
        ("theta_E_RAR_arcsec", os.path.join(out_dir, "lensing_summary_pivot_RAR.csv")),
        ("theta_E_RAR_phscaled_arcsec", os.path.join(out_dir, "lensing_summary_pivot_scaled.csv")),
        ("alpha_req_at_thetaE_obs", os.path.join(out_dir, "lensing_alpha_req_pivot.csv")),
    ]
    for field, out_path in pivots:
        if field in base_fields:
            fn, wide_rows = pivot_long_to_wide(combined_rows, key_field="lens_id", value_field=field)
            write_csv(out_path, fn, wide_rows)
            print(f"[OK] Wrote pivot for '{field}': {out_path} ({len(wide_rows)} rows)")


if __name__ == "__main__":
    main()
