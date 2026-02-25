"""Abstract base classes for all benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from numpy.typing import NDArray

from finchge.utils.random_mixin import RandomStateMixin


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark problem."""

    name: str
    category: str
    description: str
    reference: str
    input_dim: int
    output_dim: int
    train_size: Optional[int] = None
    test_size: Optional[int] = None


class Benchmark(RandomStateMixin, ABC):
    """
    Abstract base class for all benchmarks.

    A benchmark defines:
    1. The grammar for the problem
    2. The data (training and test) - optional for control problems
    3. How to evaluate fitness

    All benchmarks must be reproducible - same seed = same results.
    """

    def __init__(self, random_state: Optional[Any] = None) -> None:
        super().__init__(random_state=random_state)
        """
        Initialize benchmark.

        Args:
            random_state: Seed for reproducibility
        """
        self._train_data: Optional[Tuple[NDArray[Any], NDArray[Any]]] = None
        self._test_data: Optional[Tuple[NDArray[Any], NDArray[Any]]] = None
        self.random_state = random_state

    @property
    @abstractmethod
    def metadata(self) -> BenchmarkMetadata:
        """Return metadata about this benchmark."""
        pass

    @abstractmethod
    def grammar(self) -> str:
        """Return the BNF grammar string for this problem."""
        pass

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """
        Generate or load the dataset.

        This method is optional. Control problems that don't use traditional
        data should override this to return dummy data or raise a clear error.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)

        Raises:
            NotImplementedError: If the benchmark doesn't implement this method
                                 and doesn't override load_data()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_data() "
            f"or override load_data() for control problems"
        )

    def load_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Load data (with caching)."""
        if self._train_data is None or self._test_data is None:
            X_train, y_train, X_test, y_test = self._generate_data()
            self._train_data = (X_train, y_train)
            self._test_data = (X_test, y_test)

        train_X, train_y = self._train_data
        test_X, test_y = self._test_data
        return train_X, train_y, test_X, test_y

    def uses_data(self) -> bool:
        """
        Check if this benchmark uses traditional train/test data.

        Returns:
            True if the benchmark implements _generate_data, False for control problems
        """
        return self.__class__._generate_data is not Benchmark._generate_data

    def __repr__(self) -> str:
        return f"{self.metadata.name}(random_state={self.random_state})"
