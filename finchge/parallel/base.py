from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.fitness import GEFitnessFunction


class BaseParallelBackend(ABC):
    """Abstract parallel backend"""

    @abstractmethod
    async def evaluate_batch(
        self,
        contexts: list[dict[str, Any]],
        fitness_functions: list["GEFitnessFunction"],
    ) -> list[list[float]]:
        """Evaluate batch of individuals"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources"""
        pass
