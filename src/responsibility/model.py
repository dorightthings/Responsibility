"""Public model interface for training and evaluation entry points."""

from __future__ import annotations

from typing import Any

from .rrca import MechanismGroupedContextMASTER


ResponsibilityModel = MechanismGroupedContextMASTER


def build_model(**kwargs: Any) -> ResponsibilityModel:
    """Build CRFR + stock-specific RRCA with four-mechanism temporal memory.

    Keyword arguments are passed through unchanged to
    :class:`MechanismGroupedContextMASTER` so experiment configuration remains
    explicit. Memory has four learned coefficients, no separate learning rate,
    and the same state-dict keys as the retained memory experiments.
    """

    return ResponsibilityModel(**kwargs)


__all__ = ["ResponsibilityModel", "build_model"]
