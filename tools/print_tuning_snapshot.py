#!/usr/bin/env python3
import json
import os
import sys


def main():
    # Default path if not provided
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "your_run_id", "tuning_snapshot.json")
    if not os.path.exists(path):
        print(f"Snapshot not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        snap = json.load(f)

    print("metadata:", snap.get("metadata"))
    print("performance:", snap.get("performance"))
    print("convergence:", snap.get("convergence"))
    print("best_fit:", snap.get("best_fit"))

    posterior_stats = snap.get("posterior_stats", {}) or {}
    try:
        keys = list(posterior_stats.keys())
    except Exception:
        keys = []
    print("posterior_stats keys:", keys)

    # Handle both possible schemas for suggested_bounds
    sb = snap.get("suggested_bounds", {}) or {}
    bounds_map = sb.get("bounds") or sb.get("per_param") or {}
    # Convert mapping to items (handles list of tuples gracefully if given)
    try:
        items = list(bounds_map.items())
    except Exception:
        items = []
    print("suggested_bounds[0..2]:", items[:2])

    top_pcs = snap.get("top_pcs", []) or []
    print("top_pcs count:", len(top_pcs))

    seed_lp = snap.get("seed_live_points", {}) or {}
    print("seed_live_points K:", seed_lp.get("K"))

    # Sampler tuning suggestions
    print("sampler_tuning:", snap.get("sampler_tuning"))

    # Latest checkpoint paths (if available)
    latest = (snap.get("metadata", {}) or {}).get("latest_checkpoint")
    print("latest_checkpoint:", latest)


if __name__ == "__main__":
    main()

