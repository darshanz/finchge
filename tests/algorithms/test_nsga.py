from unittest.mock import MagicMock

from finchge.algorithm.nsga import NSGA2, NSGA3
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.operators.replacement import NSGA2Replacement


class _FakeEvaluatorMO:
    def __init__(self, n_objectives: int = 2, maximize: bool = False):
        self._flags = [maximize] * n_objectives

    def get_maximize_flags(self):
        return self._flags

    def refresh_mapping_all(self, individuals):
        pass

    def evaluate_population(self, population):
        for i, ind in enumerate(population.individuals):
            if not ind.fitness:
                ind.fitness = [float(i % 3), float((i + 1) % 3)]

    @property
    def require_case_data(self):
        return False


def _mo_ind(f0: float, f1: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3, 4, 5])
    ind.phenotype = "x"
    ind.fitness = [f0, f1]
    return ind


def _make_nsga2(n_objectives: int = 2, elite_size: int = 0) -> NSGA2:
    evaluator = _FakeEvaluatorMO(n_objectives=n_objectives)
    crossover = MagicMock()
    crossover.cross.side_effect = lambda a, b: (a.clone(), b.clone())
    mutation = MagicMock()
    mutation.mutate.side_effect = lambda ind: ind.clone()
    selection = MagicMock()
    selection.requires_case_data = False
    selection.select.side_effect = lambda population_size, individuals: individuals
    replacement = NSGA2Replacement(maximize_flags=[True, True])

    return NSGA2(
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        replacement=replacement,
        elite_size=elite_size,
        fitness_evaluator=evaluator,
        random_state=42,
    )


def test_nsga2_evolve_preserves_population_size():
    algo = _make_nsga2()
    individuals = [_mo_ind(float(i), float(4 - i)) for i in range(6)]
    pop = Population.from_individuals(individuals, population_size=6)
    new_pop = algo.evolve_one_generation(pop)
    assert len(new_pop) == 6


def test_nsga2_evolve_with_one_invalid_does_not_crash():
    """Regression for F2: invalid individual in population must not crash the generation loop."""
    algo = _make_nsga2()
    valid_individuals = [_mo_ind(float(i), float(4 - i)) for i in range(5)]
    bad = Individual.from_genotype([99])
    bad.mark_invalid()
    pop = Population.from_individuals(valid_individuals + [bad], population_size=6)
    new_pop = algo.evolve_one_generation(pop)
    assert len(new_pop) == 6


def test_nsga2_assigns_rank_meta_after_sort():
    algo = _make_nsga2()
    individuals = [_mo_ind(float(i), float(4 - i)) for i in range(4)]
    pop = Population.from_individuals(individuals, population_size=4)
    algo.sort_population(pop)
    for ind in pop.individuals:
        assert "rank" in ind.meta


def test_nsga2_assigns_crowding_distance_after_sort():
    """Regression for F3: crowding distance must be set on all fronts after sort."""
    algo = _make_nsga2()
    individuals = [_mo_ind(float(i), float(4 - i)) for i in range(4)]
    pop = Population.from_individuals(individuals, population_size=4)
    algo.sort_population(pop)
    for ind in pop.individuals:
        assert "crowding_distance" in ind.meta


def test_nsga2_is_multi_objective_base():
    from finchge.algorithm.base import BaseAlgorithmMO

    algo = _make_nsga2()
    assert isinstance(algo, BaseAlgorithmMO)


def _make_nsga3(n_objectives: int = 2) -> NSGA3:
    evaluator = _FakeEvaluatorMO(n_objectives=n_objectives)
    crossover = MagicMock()
    crossover.cross.side_effect = lambda a, b: (a.clone(), b.clone())
    mutation = MagicMock()
    mutation.mutate.side_effect = lambda ind: ind.clone()
    selection = MagicMock()
    selection.requires_case_data = False
    selection.select.side_effect = lambda population_size, individuals: individuals

    return NSGA3(
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        fitness_evaluator=evaluator,
        num_divisions=4,
        random_state=42,
    )


def test_nsga3_evolve_preserves_population_size():
    algo = _make_nsga3(n_objectives=2)
    individuals = [_mo_ind(float(i), float(6 - i)) for i in range(6)]
    pop = Population.from_individuals(individuals, population_size=6)
    new_pop = algo.evolve_one_generation(pop)
    assert len(new_pop) == 6


def test_nsga3_is_multi_objective_base():
    from finchge.algorithm.base import BaseAlgorithmMO

    algo = _make_nsga3()
    assert isinstance(algo, BaseAlgorithmMO)
