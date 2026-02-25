from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from finchge.runners.base import DataAwareRunner, DatasetProtocol


class MLModelRunner(DataAwareRunner):
    """
    Runner for problems where phenotype builds/trains a model.
    """

    def __init__(
        self,
        data_train: Union[Tuple[Any, Any], DatasetProtocol],
        data_val: Union[Tuple[Any, Any], DatasetProtocol],
        model_parser: Any,  # Should be BaseModelParser type
        data_test: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize model-building runner.

        Args:
            data_train: Training data for model fitting
            data_val: Validation data for fitness evaluation
            model_parser: Parser that converts phenotype to model
            data_test: Test data (optional)
        """
        super().__init__(
            random_state=random_state,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
        )
        self.model_parser = model_parser

    def run(
        self, phenotype: str, context_hints: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Build model, train on training data, predict on validation data.

        Args:
            phenotype: phenotype string that can be parsed as a ML model.
            context_hints: What context keys the fitness functions need.
                          None means assume minimal (y_pred/y_true only).

        Returns:
            dict with context having y_pred, y_true and other context info as requested.
        """
        X_train: Optional[NDArray[np.float64]] = None
        y_train: Optional[NDArray[np.float64]] = None
        X_val: Optional[NDArray[np.float64]] = None
        y_val: Optional[NDArray[np.float64]] = None
        model: Any = None
        y_pred: NDArray[np.float64]

        if self.data_info.get("X_type") == "torch":
            from finchge.model.model import Model

            net_ = self.model_parser.parse(phenotype)
            model = Model(net=net_)
            model.fit(train_dataset=self.data_train, val_dataset=self.data_test)
            # For torch path, predict returns (y_pred, y_true)
            y_pred, y_val = model.predict(self.data_test)
            # Ensure both are numpy arrays
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred, dtype=np.float64)
            if y_val is not None and not isinstance(y_val, np.ndarray):
                y_val = np.array(y_val, dtype=np.float64)
        else:
            # Convert data to numpy
            X_train, y_train = self._convert_to_numpy(self.data_train)
            X_val, y_val = self._convert_to_numpy(self.data_val)

            model = self.model_parser.parse(phenotype)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Ensure y_pred is numpy array
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred, dtype=np.float64)

        # Build base context
        context: Dict[str, Any] = {
            "y_pred": y_pred,
            "phenotype": phenotype,
        }

        # Add y_true if available
        if y_val is not None:
            context["y_true"] = y_val

        # If no hints, return minimal context
        if context_hints is None:
            return context

        # Add requested items
        if "model" in context_hints and model is not None:
            context["model"] = model

        if "X_train" in context_hints and X_train is not None:
            context["X_train"] = X_train

        if "y_train" in context_hints and y_train is not None:
            context["y_train"] = y_train

        if "X_val" in context_hints and X_val is not None:
            context["X_val"] = X_val

        if "feature_importance" in context_hints and hasattr(
            model, "feature_importances_"
        ):
            context["feature_importance"] = model.feature_importances_

        return context
