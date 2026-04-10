from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from finchge.runners.base import DatasetProtocol, TrainEvalRunner


class MLModelRunner(TrainEvalRunner):
    """
    Runner for problems where phenotype builds and trains a model.
    This runner can be used in problems such as hyperparameter optimization
    and architecture search etc.

    The phenotype is parsed into a trainable model or configuration,
    fitted on the training split, and evaluated on the active evaluation split.
    """

    def __init__(
        self,
        data_train: Union[Tuple[Any, Any], DatasetProtocol],
        data_val: Union[Tuple[Any, Any], DatasetProtocol],
        model_parser: Any,
        data_test: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
        )
        self.model_parser = model_parser

    def build_model(self, phenotype: str) -> Any:
        return self.model_parser.parse(phenotype)

    def fit_model(
        self,
        model: Any,
        X_train: Optional[NDArray[np.float64]],
        y_train: Optional[NDArray[np.float64]],
    ) -> Any:
        if self.data_info.get("X_type") == "torch":
            # Assumes torch-compatible model wrapper handles dataset objects
            model.fit(train_dataset=self.data_train, val_dataset=self.data_val)
        else:
            if X_train is None or y_train is None:
                raise ValueError(
                    "X_train and y_train must be provided for non-torch models"
                )
            model.fit(X_train, y_train)
        return model

    def predict_eval(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        if self.data_info.get("X_type") == "torch":
            if self.eval_split == "train":
                dataset = self.data_train
            elif self.eval_split == "val":
                dataset = self.data_val
            else:
                dataset = self.data_test

            y_pred, _ = model.predict(dataset)
            return np.array(y_pred, dtype=np.float64)

        if X_eval is None:
            raise ValueError("X_eval must be provided for non-torch models")

        y_pred = model.predict(X_eval)
        return np.array(y_pred, dtype=np.float64)

    def predict_eval_proba(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.float64]]:
        if self.data_info.get("X_type") == "torch":
            return None

        if X_eval is not None and hasattr(model, "predict_proba"):
            return np.array(model.predict_proba(X_eval), dtype=np.float64)

        return None

    def predict_eval_score(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.float64]]:
        if self.data_info.get("X_type") == "torch":
            return None

        if X_eval is not None and hasattr(model, "decision_function"):
            return np.array(model.decision_function(X_eval), dtype=np.float64)

        return None
