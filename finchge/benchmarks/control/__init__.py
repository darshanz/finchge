"""
Control benchmarks for genetic programming.

This module provides benchmark problems for control tasks:
- Santa Fe Trail (artificial ant)
- Maze navigation
- Cart-pole balancing
- And more...
"""

from finchge.benchmarks.control.base import ControlEnvironment
from finchge.benchmarks.control.santafe import SantaFeEnvironment, SantaFeTrailBenchmark

__all__ = [
    # Base classes
    "ControlEnvironment",
    # Santa Fe Trail
    "SantaFeEnvironment",
    "SantaFeTrailBenchmark",
]
