from __future__ import annotations

from typing import Callable, Tuple, TypeAlias, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

# Arrays
FloatArray: TypeAlias = NDArray[np.float64]

# Ranges
Range1D: TypeAlias = tuple[float, float]
Range2D: TypeAlias = tuple[Range1D, Range1D]
Range: TypeAlias = Range1D | Range2D

# Benchmark callable
BenchmarkCallable: TypeAlias = Callable[[FloatArray], FloatArray]


# Base function metadata
class BenchmarkFunctionInfoBase(TypedDict, total=False):
    name: str
    equation: str
    description: str
    function: BenchmarkCallable
    input_dim: int
    output_dim: int
    complexity: str
    reference: str
    default_range: Tuple[float, float]
    train_range: tuple[float, float]
    test_range: tuple[float, float]
    train_step: float | None
    test_step: float
    test_n_points: int
    note: str


class BenchmarkSpec(TypedDict):
    dim: int
    range: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]
    func: Callable[[FloatArray], FloatArray]
    name: str
