"""Domain package: lazy-load heavy modules so `domain.model` imports stay lightweight."""

from __future__ import annotations

__all__ = [
    "build_dataset",
    "compute_motion_states",
    "compute_spatial_configuration",
    "compute_ttc_risk",
    "motion_lookup_by_scene_key",
]


def __getattr__(name: str):
    if name == "build_dataset":
        from .builder import build_dataset

        return build_dataset
    if name == "motion_lookup_by_scene_key":
        from .builder import motion_lookup_by_scene_key

        return motion_lookup_by_scene_key
    if name == "compute_motion_states":
        from .relations import compute_motion_states

        return compute_motion_states
    if name == "compute_spatial_configuration":
        from .relations import compute_spatial_configuration

        return compute_spatial_configuration
    if name == "compute_ttc_risk":
        from .relations import compute_ttc_risk

        return compute_ttc_risk
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
