from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from finchge.config.config import FinchConfig, Keys
from finchge.core.engine import GrammaticalEvolution
from finchge.core.individual import Individual
from finchge.core.result import GEResult
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.grammar import GenotypeMapper, Grammar
from finchge.runners.sr import SymbolicRegressionRunner
from finchge.symbolic.expression import SymbolicExpression
from finchge.utils.logger import ExperimentLogger
from finchge.utils.random_mixin import RandomStateMixin


class GERegressor(RandomStateMixin, BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """
    Sklearn-style wrapper for symbolic regression using Grammatical Evolution.
    """

    def __init__(
        self,
        grammar: Grammar,
        config: FinchConfig | dict[str, Any],
        fitness_functions: GEFitnessFunction | list[GEFitnessFunction],
        generations: int = 100,
        population_size: int = 100,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state=random_state)
        self.grammar = grammar
        self.generations = generations
        self.population_size = population_size

        # learned attributes
        self.best_individual_: Optional[Individual] = None
        self.best_phenotype_: Optional[str] = None

        self.random_state = random_state

        self.fitness_functions = fitness_functions

        if isinstance(config, FinchConfig):
            self.config = config
        else:
            self.config = FinchConfig.from_dict(config)

        self.genoype_mapper = GenotypeMapper(
            grammar=self.grammar,
            max_wraps=self.config.ge[Keys.MAX_WRAPS],
            max_recursion_depth=self.config.ge[Keys.MAX_RECURSION_DEPTH],
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> "GERegressor":
        sym_runner = SymbolicRegressionRunner(
            data_train=(X, y),
        )
        fitness_evaluator = FitnessEvaluator(
            runner=sym_runner,
            fitness_functions=self.fitness_functions,
            mapper=self.genoype_mapper,
            parallel_config=self.config.parallel,
        )
        expt_logger = ExperimentLogger()
        ge = GrammaticalEvolution(
            grammar=self.grammar,
            fitness_evaluator=fitness_evaluator,
            config=self.config,
            expt_logger=expt_logger,
            random_state=self.random_state,
        )

        ge_result: GEResult = ge.run()

        self.ge_result_ = ge_result

        # there will be best individual if it is single objective
        self.selected_individual_ = ge_result.best_individual

        if self.selected_individual_ is not None:
            self.selected_expression_: SymbolicExpression | None = SymbolicExpression(
                self.selected_individual_.phenotype
            )
        else:
            self.selected_expression_ = None

        return self

    def select_individual(self, individual: Individual) -> None:
        self.selected_individual_ = individual
        self.selected_expression_ = SymbolicExpression(individual.phenotype)

    def predict(self, X: Any) -> NDArray[np.float64]:
        if self.selected_expression_ is None:
            raise RuntimeError("Model has not been fitted or no valid model selected")

        return self.selected_expression_.eval(X)
