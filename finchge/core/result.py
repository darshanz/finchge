from finchge.core import Individual, Population


class GEResult:
    def __init__(
        self,
        best_individual: Individual | None,
        pareto_front: list[Individual] | None,
        population: Population,
    ) -> None:
        self.best_individual = best_individual
        self.pareto_front = pareto_front
        self.population = population