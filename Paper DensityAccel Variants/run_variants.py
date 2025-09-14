# Experimental orchestrator for density vs acceleration gate variants
# This mirrors scripts/next_steps_from_run.py but exposes a clean switch
# gate_family ∈ {"density", "accel"} and gathers the same artifacts into
# a separate results/images subtree to avoid touching the main repo paths.

import argparse
import os
from pathlib import Path
import json

from xi_density_plateau import xi_density_plateau


def main():
    ap = argparse.ArgumentParser(description="Run density- vs acceleration-gated paper analyses (sandbox)")
    ap.add_argument("--gate-family", choices=["density", "accel"], default="density")
    ap.add_argument("--out-root", default="Paper DensityAccel Variants")
    # Density gate params
    ap.add_argument("--rho-c", type=float, default=1e-25)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--xi-max", type=float, default=50.0)
    # Accel gate params (legacy)
    ap.add_argument("--a0", type=float, default=1.93e-7)
    ap.add_argument("--Dmax", type=float, default=50.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Record chosen gate config for later inspection
    cfg = {
        "gate_family": args.gate_family,
        "density": {"rho_c": args.rho_c, "gamma": args.gamma, "xi_max": args.xi_max},
        "accel": {"a0": args.a0, "Dmax": args.Dmax},
    }
    (out_root / "gate_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Placeholder: here you'd import your existing per-section runners and pass
    # the gate as a callable:
    #   xi = (lambda x: xi_density_plateau(x, rho_c=args.rho_c, gamma=args.gamma, xi_max=args.xi_max))
    # or legacy accel gate xi_accel(g_bar, a0, Dmax)
    # Since we are not to modify existing modules, keep this sandbox orchestrator
    # focused on config export for now.

    print("Wrote sandbox config:", (out_root / "gate_config.json").as_posix())


if __name__ == "__main__":
    main()
