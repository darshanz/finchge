from typing import Any

import numpy as np
from numpy.random import RandomState


def generate_vlad_points(
    dim: int,
    n: int,
    x_range: Any,
    np_rng: RandomState,
) -> np.ndarray:
    pts = np.zeros((n, dim))

    for i in range(dim):
        low, high = x_range[i]
        pts[:, i] = np_rng.uniform(low, high, n)

    return pts.astype(np.float64)
