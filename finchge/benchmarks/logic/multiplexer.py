from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.registry import register
from finchge.grammar import Grammar
from finchge.runners.base import PhenotypeRunner


@register("multiplexer_6", "logic")
class Multiplexer6Benchmark(Benchmark):
    """
    6-bit multiplexer problem.
    2 address bits + 4 data bits, 64 truth table entries.
    """

    def __init__(self, random_state: Optional[Any] = None):
        super().__init__(random_state=random_state)
        self._metadata = BenchmarkMetadata(
            name="6-bit Multiplexer",
            category="logic",
            description="6-bit multiplexer: 2 address + 4 data bits",
            reference="Koza, J.R. (1992). Genetic Programming.",
            input_dim=6,
            output_dim=1,
            train_size=64,
            test_size=64,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        return """
        <expr> ::= <if> | <and> | <or> | <not> | <var>
        <if> ::= if ( <expr> , <expr> , <expr> )
        <and> ::= and ( <expr> , <expr> )
        <or>  ::= or ( <expr> , <expr> )
        <not> ::= not ( <expr> )
        <var> ::= x[0..5]
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        n_bits = 6
        n_address = 2
        n_combinations = 2**n_bits

        X = np.zeros((n_combinations, n_bits), dtype=np.int8)
        y = np.zeros(n_combinations, dtype=np.int8)

        for i in range(n_combinations):
            bits = [(i >> j) & 1 for j in range(n_bits)]
            X[i] = bits
            address = 0
            for j in range(n_address):
                address |= bits[j] << j
            y[i] = bits[n_address + address]

        return X, y, X.copy(), y.copy()

    def create_runner(self, data_type: str = "train") -> PhenotypeRunner:
        from finchge.runners.logic import LogicRunner

        X, y, _, _ = self._generate_data()
        return LogicRunner(X, y)

    def __repr__(self) -> str:
        return "Multiplexer6Benchmark(6-bit, 64 cases)"


@register("multiplexer_11", "logic")
class Multiplexer11Benchmark(Benchmark):
    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state=random_state)
        self._metadata = BenchmarkMetadata(
            name="11-bit Multiplexer",
            category="logic",
            description="11-bit multiplexer: 3 address + 8 data bits",
            reference="Koza, J.R. (1992). Genetic Programming.",
            input_dim=11,
            output_dim=1,
            train_size=2048,
            test_size=2048,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar(self) -> str:
        return """
        <expr> ::= <if> | <and> | <or> | <not> | <var>
        <if> ::= if ( <expr> , <expr> , <expr> )
        <and> ::= and ( <expr> , <expr> )
        <or>  ::= or ( <expr> , <expr> )
        <not> ::= not ( <expr> )
        <var> ::= x[0..10]
        """

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        n_bits = 11
        n_address = 3
        n_combinations = 2**n_bits

        X = np.zeros((n_combinations, n_bits), dtype=np.int8)
        y = np.zeros(n_combinations, dtype=np.int8)

        for i in range(n_combinations):
            bits = [(i >> j) & 1 for j in range(n_bits)]
            X[i] = bits
            address = 0
            for j in range(n_address):
                address |= bits[j] << j
            y[i] = bits[n_address + address]

        return X, y, X.copy(), y.copy()

    def create_runner(self, data_type: str = "train") -> PhenotypeRunner:
        from finchge.runners.logic import LogicRunner

        X, y, _, _ = self._generate_data()
        return LogicRunner(X, y)

    def __repr__(self) -> str:
        return "Multiplexer11Benchmark(11-bit, 2048 cases)"
