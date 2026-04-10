from abc import ABC, abstractmethod


class RepairStrategy(ABC):
    """Abstract base class for phenotype repair strategies."""

    @abstractmethod
    def repair(self, phenotype: str) -> str:
        """
        Repair the given phenotype and return repaired phenotype
        """
        pass
