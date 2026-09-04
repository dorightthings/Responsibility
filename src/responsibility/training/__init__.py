"""Portable training and evaluation helpers for ResponsibilityModel.

The public command-line entry points live in ``scripts/``.  This package keeps
their implementation importable so that the same code can be tested without
starting subprocesses.
"""

from .config import ExperimentConfig, load_experiment_config
from .engine import train_one_seed

__all__ = ["ExperimentConfig", "load_experiment_config", "train_one_seed"]
