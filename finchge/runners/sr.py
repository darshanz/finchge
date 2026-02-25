import logging
from typing import Any, Optional, Tuple, Union

import numpy as np

from finchge.runners.base import DataAwareRunner, DatasetProtocol
from finchge.symbolic.expression import SymbolicExpression


class SymbolicRegressionRunner(DataAwareRunner):
    """
    Runner for symbolic regression problems.
    Phenotype is a mathematical expression evaluated directly on X.
    """

    def __init__(
        self,
        data_train: Union[Tuple[Any, Any], DatasetProtocol],
        data_val: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        data_test: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize regression runner.

        Args:
            data_train: Training data as (X, y) tuple or Dataset
            data_val: Validation data (optional)
            data_test: Test data (optional)

        """
        super().__init__(
            random_state=random_state,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
        )

    def run(
        self, phenotype: str, context_hints: Optional[set[str]] = None
    ) -> dict[str, Any]:
        X, y_true = self._get_current_data()
        try:
            symbolic_expression = SymbolicExpression(phenotype)
            y_pred = symbolic_expression.eval(X=X)
        except Exception as ex:
            logging.debug(f"Exception: {ex}")
            y_pred = np.full(X.shape[0], np.nan)

        context = {"y_pred": y_pred, "y_true": y_true, "phenotype": phenotype}

        if context_hints is None:
            return context

        # Add expensive items only if hinted
        if "X" in context_hints:
            context["X"] = X

        if "y_train" in context_hints or "X_train" in context_hints:
            if "X_train" in context_hints:
                context["X_train"] = X
            if "y_train" in context_hints:
                context["y_train"] = y_true

        return context
