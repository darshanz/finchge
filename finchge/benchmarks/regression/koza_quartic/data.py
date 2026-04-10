from typing import Literal

import numpy as np
from numpy.random import RandomState


def generate_koza_points(
    n: int,
    x_range: tuple[float, float],
    sample_type: Literal["uniform", "grid", "random"],
    np_rng: RandomState,
) -> np.ndarray:
    low, high = x_range

    if sample_type in ("uniform", "random"):
        return np_rng.uniform(low, high, (n, 1)).astype(np.float64)

    if sample_type == "grid":
        return np.linspace(low, high, n).reshape(-1, 1).astype(np.float64)

    raise ValueError(f"Unknown sample type: {sample_type}")
