import argparse
import csv
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Combine multiple lensing_rar_table.csv files with run labels.")
    p.add_argument("--inputs", nargs="+", required=True, help="List of input CSV files to combine.")
    p.add_argument("--labels", nargs="+", required=True, help="List of labels (same length as inputs).")
    p.add_argument("--out", required=True, help="Output CSV file path.")
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

    write_csv(args.out, base_fields, combined_rows)
    print(f"[OK] Wrote combined lensing table: {args.out} ({len(combined_rows)} rows)")


if __name__ == "__main__":
    main()
