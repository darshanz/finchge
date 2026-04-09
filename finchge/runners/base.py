from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from finchge.utils.random_mixin import RandomStateMixin

# Type variables for flexibility
T = TypeVar("T")
XType = TypeVar("XType")
YType = TypeVar("YType")


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for PyTorch-like datasets."""

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ...

    def __len__(self) -> int:
        ...


class PhenotypeRunner(RandomStateMixin, ABC):
    """
    Base class for all phenotype runners.

    Every runner must implement run() which returns
    (predictions, targets) as numpy arrays.
    """

    def __init__(self, random_state: Optional[Any] = None):
        super().__init__(random_state=random_state)

    @abstractmethod
    def run(
        self, phenotype: str, context_hints: Optional[set[str]] = None
    ) -> dict[str, Any]:
        """
        Run phenotype and return context dictionary.

        Args:
            phenotype: Program to evaluate
            context_hints: What context keys the fitness functions need.
                          None means assume minimal (y_pred/y_true only).

        Returns:
            Dictionary with at least 'y_pred' and 'y_true', plus any hinted keys.
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """
        Optional metadata about the executor/problem.

        Returns:
            Dictionary with metadata (can be empty)
        """
        return {}


class DataAwareRunner(PhenotypeRunner, ABC):
    """
    Base class for runners that operate on split datasets.
    This class is responsible for storing train/val/test data, managing the active evaluation split, inferring
    data format/type, and converting supported formats to numpy when needed
    Subclasses define how the phenotype is evaluated using these splits.
    """

    def __init__(
        self,
        data_train: Union[Tuple[Any, Any], DatasetProtocol],
        data_val: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        data_test: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        self.data_train = data_train
        self.data_val = data_val if data_val is not None else data_train
        self.data_test = data_test if data_test is not None else self.data_val

        # Active evaluation split used by run()
        self.eval_split: str = "val"

        # Infer format from training data
        self.data_info = self._infer_data_types(data_train)

    def set_eval_split(self, split: str) -> None:
        """Set the active evaluation split."""
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be one of 'train', 'val', or 'test', got {split}"
            )
        self.eval_split = split

    def _infer_data_types(self, data: Any) -> dict[str, str]:
        """
        Infer the input format and X container type.
        """
        result: dict[str, str] = {}

        # Dataset-like object
        if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
            try:
                sample = data[0]
                if isinstance(sample, tuple) and len(sample) == 2:
                    result["format"] = "dataset"
                    result["X_type"] = "torch"
                    return result
            except Exception:
                pass

        # Tuple format: (X, y)
        if isinstance(data, tuple) and len(data) == 2:
            result["format"] = "tuple"
            X, _ = data

            if hasattr(X, "__class__") and X.__class__.__name__ == "DataFrame":
                result["X_type"] = "pandas"
            elif isinstance(X, np.ndarray):
                result["X_type"] = "numpy"
            elif hasattr(X, "numpy"):
                result["X_type"] = "torch"
            else:
                result["X_type"] = "unknown"

            return result

        raise ValueError(f"Unsupported data format: {type(data)}")

    def _convert_to_numpy(
        self, data: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert supported data formats to numpy arrays.
        """
        data_info = self._infer_data_types(data)

        if data_info["format"] == "dataset":
            X_list: list[NDArray[np.float64]] = []
            y_list: list[NDArray[np.float64]] = []

            for i in range(len(data)):
                x_item, y_item = data[i]

                if hasattr(x_item, "numpy"):
                    x_item = x_item.numpy()
                if hasattr(y_item, "numpy"):
                    y_item = y_item.numpy()

                X_list.append(np.array(x_item, dtype=np.float64))
                y_list.append(np.array(y_item, dtype=np.float64))

            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)

            if y.ndim > 1 and y.shape[1] == 1:
                y = y.ravel()

            return X, y

        # Tuple format
        X, y = data

        if hasattr(X, "values"):  # pandas
            X = X.values
        elif hasattr(X, "numpy"):  # torch tensor
            X = X.numpy()

        if hasattr(y, "values"):
            y = y.values
        elif hasattr(y, "numpy"):
            y = y.numpy()

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        if y_arr.ndim > 1 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()

        return X_arr, y_arr

    def get_split_data(
        self, split: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the requested split as numpy arrays.
        """
        if split == "train":
            return self._convert_to_numpy(self.data_train)
        if split == "val":
            return self._convert_to_numpy(self.data_val)
        if split == "test":
            return self._convert_to_numpy(self.data_test)
        raise ValueError(f"Unknown split '{split}'")

    def get_eval_data(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the active evaluation split as numpy arrays.
        """
        return self.get_split_data(self.eval_split)

    @property
    def provided_context_keys(self) -> set[str]:
        """
        The context keys available for data related runners
        """
        return {
            "phenotype",
            "X",
            "X_eval",
            "y_eval",
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
        }


class DirectEvalRunner(DataAwareRunner, ABC):
    """
    Base runner for direct evaluation.
    Intended for problems where the phenotype already represents the predictive
    function and does not need to be trained or fitted first. For example, symbolic regression
    """

    @abstractmethod
    def predict_direct(
        self,
        phenotype: str,
        X_eval: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Produce predictions from phenotype on the active eval split.
        """
        raise NotImplementedError

    def run(
        self,
        phenotype: str,
        context_hints: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        context_hints = context_hints or set()

        X_eval, y_eval = self.get_eval_data()

        try:
            y_pred = self.predict_direct(phenotype=phenotype, X_eval=X_eval)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred, dtype=np.float64)
        except Exception:
            y_pred = np.full(X_eval.shape[0], np.nan, dtype=np.float64)

        context: Dict[str, Any] = {
            "phenotype": phenotype,
            "y_true": y_eval,
            "y_pred": y_pred,
        }

        if "X" in context_hints or "X_eval" in context_hints:
            context["X"] = X_eval
            context["X_eval"] = X_eval

        if "y_eval" in context_hints:
            context["y_eval"] = y_eval

        if "X_train" in context_hints or "y_train" in context_hints:
            X_train, y_train = self.get_split_data("train")
            if "X_train" in context_hints:
                context["X_train"] = X_train
            if "y_train" in context_hints:
                context["y_train"] = y_train

        if "X_val" in context_hints or "y_val" in context_hints:
            X_val, y_val = self.get_split_data("val")
            if "X_val" in context_hints:
                context["X_val"] = X_val
            if "y_val" in context_hints:
                context["y_val"] = y_val

        if "X_test" in context_hints or "y_test" in context_hints:
            X_test, y_test = self.get_split_data("test")
            if "X_test" in context_hints:
                context["X_test"] = X_test
            if "y_test" in context_hints:
                context["y_test"] = y_test

        return context

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {
            "y_true",
            "y_pred",
        }


class TrainEvalRunner(DataAwareRunner, ABC):
    """
    Base runner for train-evaluate based runners. For problems where the phenotype defines a trainable model,
    architecture, or hyperparameter configuration. For example,  hyperparameter optimization, architecture search,
    evolving sklearn or pytorch model configurations.
    """

    @abstractmethod
    def build_model(self, phenotype: str) -> Any:
        """
        Build or parse a model or model configuration from phenotype.
        will be used by runners to build a model based on model parsers
        """
        raise NotImplementedError

    @abstractmethod
    def fit_model(
        self,
        model: Any,
        X_train: Optional[NDArray[np.float64]],
        y_train: Optional[NDArray[np.float64]],
    ) -> Any:
        """
        Train - fit the model on the training split.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_eval(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Predict on the active evaluation split.
        """
        raise NotImplementedError

    def predict_eval_proba(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.float64]]:
        """
        Optional hook for probability prediction.
        """
        return None

    def predict_eval_score(
        self,
        model: Any,
        X_eval: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.float64]]:
        """
        Hook for raw score, decision function output.
        """
        return None

    def run(
        self,
        phenotype: str,
        context_hints: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        context_hints = context_hints or set()

        model: Any = None
        X_train: Optional[NDArray[np.float64]] = None
        y_train: Optional[NDArray[np.float64]] = None
        X_eval: Optional[NDArray[np.float64]] = None
        y_eval: Optional[NDArray[np.float64]] = None

        model = self.build_model(phenotype)

        if self.data_info.get("X_type") == "torch":
            # Subclass handles dataset-based training internally if needed
            model = self.fit_model(model, X_train=None, y_train=None)

            # Still expose eval truth via numpy if needed
            X_eval, y_eval = self.get_eval_data()
            y_pred = self.predict_eval(model, X_eval=None)
        else:
            X_train, y_train = self.get_split_data("train")
            X_eval, y_eval = self.get_eval_data()

            model = self.fit_model(model, X_train=X_train, y_train=y_train)
            y_pred = self.predict_eval(model, X_eval=X_eval)

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred, dtype=np.float64)

        context: Dict[str, Any] = {
            "phenotype": phenotype,
            "y_pred": y_pred,
        }

        if y_eval is not None:
            context["y_true"] = y_eval

        if "model" in context_hints:
            context["model"] = model

        if "X" in context_hints or "X_eval" in context_hints:
            if X_eval is None and self.data_info.get("X_type") != "torch":
                X_eval, _ = self.get_eval_data()
            if X_eval is not None:
                context["X"] = X_eval
                context["X_eval"] = X_eval

        if "y_eval" in context_hints and y_eval is not None:
            context["y_eval"] = y_eval

        if "X_train" in context_hints or "y_train" in context_hints:
            if X_train is None or y_train is None:
                X_train, y_train = self.get_split_data("train")
            if "X_train" in context_hints:
                context["X_train"] = X_train
            if "y_train" in context_hints:
                context["y_train"] = y_train

        if "X_val" in context_hints or "y_val" in context_hints:
            X_val, y_val = self.get_split_data("val")
            if "X_val" in context_hints:
                context["X_val"] = X_val
            if "y_val" in context_hints:
                context["y_val"] = y_val

        if "X_test" in context_hints or "y_test" in context_hints:
            X_test, y_test = self.get_split_data("test")
            if "X_test" in context_hints:
                context["X_test"] = X_test
            if "y_test" in context_hints:
                context["y_test"] = y_test

        if "y_pred_proba" in context_hints:
            y_pred_proba = self.predict_eval_proba(model, X_eval)
            if y_pred_proba is not None:
                if not isinstance(y_pred_proba, np.ndarray):
                    y_pred_proba = np.array(y_pred_proba, dtype=np.float64)
                context["y_pred_proba"] = y_pred_proba

        if "y_pred_score" in context_hints:
            y_pred_score = self.predict_eval_score(model, X_eval)
            if y_pred_score is not None:
                if not isinstance(y_pred_score, np.ndarray):
                    y_pred_score = np.array(y_pred_score, dtype=np.float64)
                context["y_pred_score"] = y_pred_score

        if "feature_importance" in context_hints and hasattr(
            model, "feature_importances_"
        ):
            context["feature_importance"] = model.feature_importances_

        return context

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {
            "y_true",
            "y_pred",
            "y_pred_proba",
            "y_pred_score",
            "model",
            "feature_importance",
        }
