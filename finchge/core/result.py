from finchge.core import Individual, Population


class GEResult:
    def __init__(
        self,
        best_in_generation: Individual | None,
        all_time_best: Individual | None,
        pareto_front: list[Individual] | None,
        population: Population,
    ) -> None:
        self.best_in_generation = best_in_generation
        self.all_time_best = all_time_best
        self.pareto_front = pareto_front
        self.population = population
