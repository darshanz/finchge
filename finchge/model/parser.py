from abc import ABC, abstractmethod
from typing import Any


class BaseModelParser(ABC):
    """
    Base class for ModelParser.

    This abstract class defines the interface for model parser to be used in tasks involving ML models
    Subclasses must implement the `parse` method.

    Args:
        maximize (bool): Indicates whether the goal is to maximize (True)
                         or minimize (False) the fitness score.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def parse(self, phenotype: str) -> Any:
        """
        Parse the phenotype into a model.

        Args:
            phenotype (str): Phenotype representation of the model.

        Returns:
            model (Any): Any model parsed from phenotype. Currently supported models are Pytorch Modules or Scikit-learn Estimators.
        """
        pass
