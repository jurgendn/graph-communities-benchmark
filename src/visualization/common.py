"""
Common utilities shared across fetch and merge modules.
"""
from typing import Any, Optional


def to_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None if unable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_params(params: Any) -> dict:
    """
    Normalize parameters from various formats into a simple dict.
    Supports both dict and list-of-objects formats from Comet API.
    """
    if isinstance(params, dict):
        return params

    out = {}
    if isinstance(params, list):
        for p in params:
            if not isinstance(p, dict):
                continue
            name = p.get("name") or p.get("key")
            val = None
            for k in ("valueCurrent", "value", "valueMax", "valueMin"):
                if k in p and p[k] is not None:
                    val = p[k]
                    break
            if name:
                out[name] = val
    return out


def extract_algorithm_name(params: dict, run_index: int = 0) -> str:
    """
    Extract algorithm name from normalized parameters.
    Falls back to run_name, id, experiment_name, or generic exp_N.
    """
    candidates = (
        "algorithm",
        "algo",
        "method",
        "name",
        "algorithm_name",
        "alg",
    )
    for cand in candidates:
        v = params.get(cand)
        if v:
            return str(v)

    fallback_candidates = ("run_name", "id", "experiment_name")
    for cand in fallback_candidates:
        v = params.get(cand)
        if v:
            return str(v)

    return f"exp_{run_index}"
