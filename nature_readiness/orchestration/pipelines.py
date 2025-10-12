"""Pipelines (preview-only scaffold)
No side effects; builds a human-readable plan for the requested bundle.
"""
from __future__ import annotations
from typing import Dict, Any


def run_pipeline_preview(command: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Return a plan dict for the selected pipeline command.
    This module intentionally performs no IO or computation.
    """
    plan: Dict[str, Any] = {"command": command, "steps": []}

    if command == "theory":
        if options.get("all"):
            plan["steps"].extend(["bianchi_checks", "stability_checks", "cT_checks", "equivalence_checks"])
        else:
            plan["steps"].append("theory_subset")

    elif command == "solar":
        if options.get("all"):
            plan["steps"].extend(["ppn", "cassini_shapiro", "llr_nordtvedt", "ephemeris_perturbations"])
        else:
            plan["steps"].append("solar_subset")

    elif command == "lensing":
        if options.get("strong"):
            plan["steps"].append("strong_lensing")
        if options.get("weak"):
            plan["steps"].append("weak_lensing")

    elif command == "dynamics":
        if options.get("sparc"):
            plan["steps"].append("sparc_subset_compare:" + ",".join(options.get("compare") or []))

    elif command == "clusters":
        if options.get("bullet"):
            plan["steps"].append("bullet_cluster")

    elif command == "cosmology":
        if options.get("compressed"):
            plan["steps"].append("compressed_cmb_bao")

    elif command == "bayes":
        if options.get("evidence"):
            plan["steps"].append("evidence")
        if options.get("kfold") is not None:
            plan["steps"].append(f"kfold:{options['kfold']}")
        if options.get("ppc"):
            plan["steps"].append("ppc")
        if options.get("sbc"):
            plan["steps"].append("sbc")

    elif command == "all":
        if options.get("fast"):
            plan["steps"].extend([
                "theory:minimal", "solar:ppn+cassini", "lensing:strong:1", "dynamics:sparc:mini", "clusters:bullet:mini"
            ])
        else:
            plan["steps"].extend([
                "theory:all", "solar:all", "lensing:strong+weak", "dynamics:sparc", "clusters:bullet", "cosmology:compressed", "bayes:evidence+ppc+sbc"
            ])

    return plan

