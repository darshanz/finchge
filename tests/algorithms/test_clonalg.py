import pytest

from finchge.algorithm.clonalg import CLONALG


class _FakeEval:
    def get_maximize_flags(self):
        return [False]

    def refresh_mapping_all(self, individuals):
        pass

    def evaluate_population(self, population):
        for ind in population.individuals:
            if ind.fitness is None:
                ind.fitness = [0.0]


def _make_algorithm(**kwargs) -> CLONALG:
    from unittest.mock import MagicMock

    from finchge.core.individual import Individual

    mutation = MagicMock()

    def _offspring():
        o = Individual.from_genotype([1])
        o.phenotype = "x"
        o.fitness = [0.0]
        return o

    mutation.mutate.side_effect = lambda ind: _offspring()
    defaults = dict(
        num_select=3,
        num_clones=5,
        hyper_factor=0.5,
        mutation=mutation,
        fitness_evaluator=_FakeEval(),
        random_state=42,
    )
    defaults.update(kwargs)
    return CLONALG(**defaults)


def test_num_select_zero_raises():
    with pytest.raises(ValueError, match="num_select"):
        _make_algorithm(num_select=0)


def test_num_clones_zero_raises():
    with pytest.raises(ValueError, match="num_clones"):
        _make_algorithm(num_clones=0)


def test_hyper_factor_zero_raises():
    with pytest.raises(ValueError, match="hyper_factor"):
        _make_algorithm(hyper_factor=0.0)


def test_hyper_factor_negative_raises():
    with pytest.raises(ValueError, match="hyper_factor"):
        _make_algorithm(hyper_factor=-1.0)


def test_max_best_set_from_evaluator():
    algo = _make_algorithm()
    assert algo.max_best is False


def test_next_generation_has_correct_size():
    from finchge.core.individual import Individual
    from finchge.core.population import Population

    algo = _make_algorithm(num_select=3, num_clones=4)
    pop_size = 10
    individuals = [Individual.from_genotype([i]) for i in range(pop_size)]
    for i, ind in enumerate(individuals):
        ind.phenotype = "x"
        ind.fitness = [float(i)]
    pop = Population.from_individuals(individuals, population_size=pop_size)
    next_pop = algo.evolve_one_generation(pop)
    assert len(next_pop.individuals) == pop_size
