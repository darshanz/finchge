from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from finchge.utils.random_mixin import RandomStateMixin

if TYPE_CHECKING:
    from finchge.core.individual import Individual


class GESelectionStrategy(RandomStateMixin, ABC):
    requires_case_data: bool = False
    required_case_keys: tuple[str, ...] = ()

    def __init__(self, max_best: bool, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        if max_best is None:
            raise ValueError("max_best parameter is required")
        self.max_best = max_best

    @abstractmethod
    def select(
        self, population_size: int, individuals: list["Individual"]
    ) -> list["Individual"]: ...


class GECrossoverStrategy(RandomStateMixin, ABC):
    """
    Abstract method for crossover implementation.

    Initialize crossover strategy.

    Args:
        crossover_proba (float): Probability of crossover occurring
    """

    def __init__(
        self, crossover_proba: float, random_state: Optional[int] = None
    ) -> None:
        super().__init__(random_state=random_state)
        if not 0 <= crossover_proba <= 1:
            raise ValueError("crossover_proba must be between 0 and 1")

        self.crossover_proba = crossover_proba

    @abstractmethod
    def cross(
        self,
        parent1: "Individual",
        parent2: "Individual",
    ) -> tuple[Individual, Individual]:
        """
        Abstract method for crossover implementation.

        Args:
            parent1: First parent
            parent2: Second parent
            within_used: If True, crossover points will be within used section
        """
        ...


class GEMutationStrategy(RandomStateMixin, ABC):
    """Base class for mutation strategies"""

    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        ...

    @abstractmethod
    def mutate(self, individual: "Individual") -> "Individual":
        """
        Abstract method for mutation implementation.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        ...


class GEReplacementStrategy(RandomStateMixin, ABC):
    """
    Base class for replacement strategy.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        ...

    @abstractmethod
    def replace(
        self,
        new_population: list["Individual"],
        old_population: list["Individual"],
        elite_size: int,
        population_size: int,
    ) -> list["Individual"]: ...
