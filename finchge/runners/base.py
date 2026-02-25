from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

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

    def __getitem__(self, idx: int) -> Tuple[Any, Any]: ...

    def __len__(self) -> int: ...


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
    Base class for runners that work with data (regression, model-building).
    Handles conversion from various data formats to numpy arrays.
    """

    def __init__(
        self,
        data_train: Union[Tuple[Any, Any], DatasetProtocol],
        data_val: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        data_test: Optional[Union[Tuple[Any, Any], DatasetProtocol]] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize data-aware executor.

        Args:
            data_train: Training data as (X, y) tuple or Dataset
            data_val: Validation data (optional)
            data_test: Test data (optional)
        """
        super().__init__(random_state=random_state)
        self.data_train = data_train
        self.data_val = data_val or data_train
        self.data_test = data_test or self.data_val

        self.mode: str = "val"  # 'train', 'val', or 'test'

        # Infer data types during initialization
        self.data_info = self._infer_data_types(data_train)

    def set_mode(self, mode: str) -> None:
        """Switch between 'train', 'val', and 'test' modes."""
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")
        self.mode = mode

    def _infer_data_types(self, data: Any) -> dict[str, str]:
        """
        Infer data types from input data.

        Returns:
            Dictionary with 'format' and 'X_type' keys
        """
        result: dict[str, str] = {}

        # Check for Dataset protocol
        if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
            try:
                # Try to see if it behaves like a dataset
                sample = data[0]
                if isinstance(sample, tuple) and len(sample) == 2:
                    result["format"] = "dataset"
                    result["X_type"] = "torch"
                    return result
            except Exception:
                pass

        # Check for tuple format
        if isinstance(data, tuple) and len(data) == 2:
            result["format"] = "tuple"
            X, _ = data

            # Check X type
            if hasattr(X, "__class__") and X.__class__.__name__ == "DataFrame":
                result["X_type"] = "pandas"
            elif isinstance(X, np.ndarray):
                result["X_type"] = "numpy"
            elif hasattr(X, "numpy"):  # PyTorch tensor
                result["X_type"] = "torch"
            else:
                result["X_type"] = "unknown"

            return result

        raise ValueError(f"Unsupported data format: {type(data)}")

    def _convert_to_numpy(
        self, data: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert any supported data format to numpy arrays.

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        data_info = self._infer_data_types(data)

        if data_info["format"] == "dataset":
            # Convert Dataset to numpy
            X_list: list[NDArray[np.float64]] = []
            y_list: list[NDArray[np.float64]] = []

            for i in range(len(data)):
                x_item, y_item = data[i]

                # Convert to numpy if needed
                if hasattr(x_item, "numpy"):
                    x_item = x_item.numpy()
                if hasattr(y_item, "numpy"):
                    y_item = y_item.numpy()

                X_list.append(np.array(x_item, dtype=np.float64))
                y_list.append(np.array(y_item, dtype=np.float64))

            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)

            # Handle case where y is 2D but should be 1D
            if y.ndim > 1 and y.shape[1] == 1:
                y = y.ravel()

            return X, y

        else:  # tuple format
            X, y = data

            # Convert X to numpy
            if hasattr(X, "values"):  # pandas DataFrame/Series
                X = X.values
            elif hasattr(X, "numpy"):  # torch tensor
                X = X.numpy()

            # Convert y to numpy
            if hasattr(y, "values"):
                y = y.values
            elif hasattr(y, "numpy"):
                y = y.numpy()

            X_arr = np.array(X, dtype=np.float64)
            y_arr = np.array(y, dtype=np.float64)

            # Ensure y is 1D
            if y_arr.ndim > 1 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()

            return X_arr, y_arr

    def _get_current_data(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get data for current mode as numpy arrays.
        """
        if self.mode == "train":
            data = self.data_train
        elif self.mode == "val":
            data = self.data_val
        else:  # test
            data = self.data_test

        return self._convert_to_numpy(data)
