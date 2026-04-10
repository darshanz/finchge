from typing import Tuple, cast

import numpy as np
from numpy.random import RandomState

from finchge.benchmarks.typing import FloatArray, Range


def generate_nguyen_points(
    input_dim: int,
    n_samples: int,
    x_range: Range,
    sample_type: str,
    np_rng: RandomState,
) -> FloatArray:
    if input_dim == 1:
        # 1D case
        range_1d = cast(Tuple[float, float], x_range)
        low, high = range_1d

        if sample_type in ["uniform", "random"]:
            points = np_rng.uniform(low, high, (n_samples, 1))
            return np.asarray(points, dtype=np.float64)
        elif sample_type == "grid":
            points = np.linspace(low, high, n_samples).reshape(-1, 1)
            return np.asarray(points, dtype=np.float64)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    else:
        # 2D case
        range_2d = cast(Tuple[Tuple[float, float], Tuple[float, float]], x_range)
        (low1, high1), (low2, high2) = range_2d

        if sample_type in ["uniform", "random"]:
            x1 = np_rng.uniform(low1, high1, n_samples)
            x2 = np_rng.uniform(low2, high2, n_samples)
            points = np.column_stack([x1, x2])
            return np.asarray(points, dtype=np.float64)

        elif sample_type == "grid":
            # Calculate grid size that gives at least n_samples
            n_per_dim = int(np.ceil(np.sqrt(n_samples)))
            x1 = np.linspace(low1, high1, n_per_dim)
            x2 = np.linspace(low2, high2, n_per_dim)
            X1, X2 = np.meshgrid(x1, x2)
            points = np.column_stack([X1.ravel(), X2.ravel()])
            # If more points than needed, randomly sample them
            if len(points) > n_samples:
                idx = np_rng.choice(len(points), n_samples, replace=False)
                points = points[idx]
            return np.asarray(points, dtype=np.float64)

        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
