import logging
from typing import Any, Dict, Optional, Set

import numpy as np
from numpy.typing import NDArray

from finchge.runners.base import DirectEvalRunner
from finchge.symbolic.expression import SymbolicExpression


class SymbolicRegressionRunner(DirectEvalRunner):
    """
    Runner for symbolic regression.

    The phenotype is interpreted as a symbolic expression and evaluated
    directly on the active evaluation split.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_saturated = False

    def predict_direct(
        self,
        phenotype: str,
        X_eval: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._last_saturated = False
        try:
            symbolic_expression = SymbolicExpression(phenotype)

            y_pred = symbolic_expression.eval(X=X_eval)
            self._last_saturated = symbolic_expression.saturated
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred, dtype=np.float64)
            return y_pred
        except Exception as ex:
            logging.debug(f"Exception while evaluating phenotype {phenotype!r}: {ex}")
            return np.full(X_eval.shape[0], np.nan, dtype=np.float64)

    def run(
        self,
        phenotype: str,
        context_hints: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        context = super().run(phenotype=phenotype, context_hints=context_hints)
        context["saturated"] = self._last_saturated
        return context
