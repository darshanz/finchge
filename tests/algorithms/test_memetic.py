import pytest

from finchge.algorithm.memetic import MemeticGA


def test_local_search_steps_zero_raises():
    from unittest.mock import MagicMock

    class _FakeEval:
        def get_maximize_flags(self):
            return [False]

    with pytest.raises(ValueError, match="local_search_steps"):
        MemeticGA(
            selection=MagicMock(),
            crossover=MagicMock(),
            mutation=MagicMock(),
            replacement=MagicMock(),
            elite_size=1,
            fitness_evaluator=_FakeEval(),
            local_search_steps=0,
        )


def test_local_search_probability_out_of_range_raises():
    from unittest.mock import MagicMock

    class _FakeEval:
        def get_maximize_flags(self):
            return [False]

    with pytest.raises(ValueError, match="local_search_probability"):
        MemeticGA(
            selection=MagicMock(),
            crossover=MagicMock(),
            mutation=MagicMock(),
            replacement=MagicMock(),
            elite_size=1,
            fitness_evaluator=_FakeEval(),
            local_search_probability=1.5,
        )


def test_max_best_set_from_evaluator():
    from unittest.mock import MagicMock

    class _FakeEval:
        def get_maximize_flags(self):
            return [True]

    algo = MemeticGA(
        selection=MagicMock(),
        crossover=MagicMock(),
        mutation=MagicMock(),
        replacement=MagicMock(),
        elite_size=1,
        fitness_evaluator=_FakeEval(),
    )
    assert algo.max_best is True


def test_is_improvement_minimize():
    from unittest.mock import MagicMock

    from finchge.core.individual import Individual

    class _FakeEval:
        def get_maximize_flags(self):
            return [False]

    algo = MemeticGA(
        selection=MagicMock(),
        crossover=MagicMock(),
        mutation=MagicMock(),
        replacement=MagicMock(),
        elite_size=1,
        fitness_evaluator=_FakeEval(),
    )
    better = Individual.from_genotype([1])
    better.phenotype = "x"
    better.fitness = [1.0]
    worse = Individual.from_genotype([2])
    worse.phenotype = "x"
    worse.fitness = [5.0]
    assert algo._is_improvement(better, worse) is True
    assert algo._is_improvement(worse, better) is False
