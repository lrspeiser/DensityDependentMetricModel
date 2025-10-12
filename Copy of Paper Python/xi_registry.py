#!/usr/bin/env python3
"""
xi_registry.py — Single source of truth for which xi models are "published"
(i.e., part of the reproducible paper pipeline) vs. "experimental".

Published = EXACTLY: 'gr' and 'tidal_band'
Experimental = everything else (off by default)
"""
from typing import Callable, Dict

# A tiny record so we can attach docs and flags per xi
class XiSpec:
    def __init__(self, fn: Callable, published: bool, doc: str):
        self.fn = fn
        self.published = published
        self.doc = doc

# We fill this registry from density_metric_cupy at import-time,
# to avoid circular imports we only create the mapping keys here.
REGISTRY: Dict[str, XiSpec] = {}

def register_xi(name: str, fn: Callable, published: bool, doc: str):
    """Register an xi implementation."""
    REGISTRY[name] = XiSpec(fn=fn, published=published, doc=doc)

def get_allowed_xi_names(allow_experimental: bool = False):
    """Return the set of xi names permitted for this run."""
    if allow_experimental:
        return sorted(REGISTRY.keys())
    return sorted([k for k, v in REGISTRY.items() if v.published])

def resolve_xi(name: str, allow_experimental: bool = False) -> Callable:
    """
    Return the xi function by name, enforcing publication status unless
    allow_experimental=True.
    """
    if name not in REGISTRY:
        raise ValueError(f"[xi_registry] Unknown xi '{name}'. Allowed: {get_allowed_xi_names(allow_experimental)}")
    spec = REGISTRY[name]
    if not spec.published and not allow_experimental:
        raise ValueError(
            f"[xi_registry] xi '{name}' is EXPERIMENTAL and is disabled for reproducible runs.\n"
            f"Allowed for paper: {get_allowed_xi_names(False)}\n"
            f"Re-run with --allow_experimental to enable."
        )
    return spec.fn

