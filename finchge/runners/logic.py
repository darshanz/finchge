from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.logic.interpreters import LogicInterpreter
from finchge.runners.base import PhenotypeRunner


class LogicRunner(PhenotypeRunner):
    """
    Runner for logic problems like multiplexer.

    Evaluates phenotype on all truth table combinations.
    Returns predictions and true values for fitness calculation.
    """

    def __init__(
        self,
        X: NDArray[np.int8],
        y: NDArray[np.int8],
        interpreter: Optional[Any] = None,  # Logic interpreter
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        """
        Initialize logic runner.

        Args:
            X: Input combinations (shape: [n_combinations, n_bits])
            y: True output values (shape: [n_combinations])
            interpreter: Program interpreter for logic language
        """

        self.X = X
        self.y = y
        self.interpreter = interpreter or LogicInterpreter()
        self.n_combinations = len(X)

    def run(
        self, phenotype: str, context_hints: Optional[set[str]] = None
    ) -> dict[str, Any]:
        """
        Evaluate phenotype on all input combinations.

        Args:
            context_hints: What context keys the fitness functions need.
                          None means assume minimal (y_pred/y_true only).
            phenotype: Logic program string

        Returns:
            dict with context having y_pred, y_true and  other context info as requested in context_hints argument
        """
        predictions = []

        for i in range(self.n_combinations):
            inputs = self.X[i]
            try:
                # Evaluate program with these inputs
                result = self.interpreter.evaluate(phenotype, inputs)
                predictions.append(float(result))
            except Exception:
                # If evaluation fails, assume wrong output
                predictions.append(0.0)

        context = {
            "y_pred": np.array(predictions, dtype=np.float64),
            "y_true": self.y.astype(np.float64),
            "phenotype": phenotype,
        }

        if context_hints is None:
            return context

        #  expensive items only if hinted
        if "X" in context_hints:
            context["X"] = self.X

        return context

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_combinations": self.n_combinations,
            "n_bits": self.X.shape[1],
            "problem": "multiplexer",
        }

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {"y_pred", "y_true"}
