from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def calc_keijzer_size(
    r: Tuple[float, float], step: Optional[float], default: int
) -> int:
    # If a step is provided (like 0.1), the number of points depends on the range
    if step is not None:
        return int(round((r[1] - r[0]) / step)) + 1
    return default


def generate_keijzer_points(
    dim: int,
    size: int,
    r: Tuple[float, float],
    step: Optional[float],
    grid: bool,
    np_rng: RandomState,
) -> NDArray[np.float64]:
    if dim == 1:
        if step is not None:
            # Fixed step sampling (deterministic)
            return np.linspace(r[0], r[1], size).reshape(-1, 1).astype(np.float64)
        # Uniform sampling (stochastic)
        return np_rng.uniform(r[0], r[1], (size, 1)).astype(np.float64)
    else:
        # 2D sampling logic
        if grid:
            n_side = int(np.ceil(np.sqrt(size)))
            line = np.linspace(r[0], r[1], n_side)
            x1, x2 = np.meshgrid(line, line)
            pts = np.column_stack([x1.ravel(), x2.ravel()])
            return pts[:size].astype(np.float64)
        return np_rng.uniform(r[0], r[1], (size, 2)).astype(np.float64)
