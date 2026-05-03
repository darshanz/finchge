from finchge.algorithm.base import BaseAlgorithm
from finchge.core.individual import Individual


class DummyAlgorithm(BaseAlgorithm):
    def sort_population(self, population):
        pass

    def get_best_individual(self, population):
        raise NotImplementedError

    def get_pareto_front(self, population):
        raise NotImplementedError

    def evolve_one_generation(self, population):
        raise NotImplementedError


class PairCrossover:
    def cross(self, parent1, parent2):
        return Individual.from_genotype([1]), Individual.from_genotype([2])


def test_apply_crossover_returns_requested_odd_size():
    algorithm = DummyAlgorithm(random_state=42)
    parents = [Individual.from_genotype([0]), Individual.from_genotype([1])]

    children = algorithm.apply_crossover(
        crossover_strategy=PairCrossover(),
        selected_individuals=parents,
        new_pop_size=5,
    )

    assert len(children) == 5


def test_apply_crossover_returns_requested_even_size():
    algorithm = DummyAlgorithm(random_state=42)
    parents = [Individual.from_genotype([0]), Individual.from_genotype([1])]

    children = algorithm.apply_crossover(
        crossover_strategy=PairCrossover(),
        selected_individuals=parents,
        new_pop_size=6,
    )

    assert len(children) == 6
