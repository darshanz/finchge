from typing import Tuple

import numpy as np


def generate_multiplexer_table(
    n_bits: int, n_address: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_combinations = 2**n_bits
    X = np.zeros((n_combinations, n_bits), dtype=np.int8)
    y = np.zeros(n_combinations, dtype=np.int8)

    #  populate the table using bit-shifting
    indices = np.arange(n_combinations, dtype=np.uint32)
    for j in range(n_bits):
        X[:, j] = (indices >> j) & 1

    # The address is the decimal value of the address bits
    address = 0
    for j in range(n_address):
        address |= X[:, j].astype(np.uint32) << j  # type: ignore

    # y is the value of the data bit at the computed address
    # Data bits start after address bits
    for i in range(n_combinations):
        y[i] = X[i, n_address + address[i]]  # type: ignore

    return X, y
