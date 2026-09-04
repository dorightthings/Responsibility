"""Public model interface for training and evaluation entry points."""

from __future__ import annotations

from typing import Any

from .rrca import MechanismGroupedContextMASTER


ResponsibilityModel = MechanismGroupedContextMASTER


def build_model(**kwargs: Any) -> ResponsibilityModel:
    """Build the retained CRFR + stock-specific RRCA model.

    Keyword arguments are passed through unchanged to
    :class:`MechanismGroupedContextMASTER` so experiment configuration remains
    explicit and checkpoint-compatible.
    """

    return ResponsibilityModel(**kwargs)


__all__ = ["ResponsibilityModel", "build_model"]
