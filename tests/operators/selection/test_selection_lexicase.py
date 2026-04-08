import pytest

from finchge.core.individual import Individual
from finchge.operators.selection import LexicaseSelection, EpsilonLexicaseSelection


def make_individual(name: str, case_values: list[float], fitness: float = 0.0) -> Individual:
    ind = Individual(phenotype=name)
    ind.fitness = [fitness]
    ind.set_meta(Individual.CASE_DATA_META_KEY, {"errors": case_values})
    return ind


def test_lexicase_selects_case_elite_individual():
    # selects only elites from the first surviving cases
    selector = LexicaseSelection(case_key="errors", case_max_best=False, random_state=123)

    a = make_individual("a", [0.0, 10.0])
    b = make_individual("b", [1.0, 0.0])
    c = make_individual("c", [2.0, 2.0])

    selected = selector.select(population_size=20, individuals=[a, b, c])

    # With these two cases, only a or b should ever survive depending on shuffled case order.
    assert all(ind.phenotype in {"a", "b"} for ind in selected)
    assert all(ind.phenotype != "c" for ind in selected)



def test_lexicase_returns_requested_number_of_individuals():
    # returns multiple parents with correct population size
    selector = LexicaseSelection(case_key="errors", case_max_best=False, random_state=42)

    individuals = [
        make_individual("a", [0.0, 1.0]),
        make_individual("b", [1.0, 0.0]),
        make_individual("c", [2.0, 2.0]),
    ]

    selected = selector.select(population_size=7, individuals=individuals)

    assert len(selected) == 7

def test_lexicase_supports_case_maximization():
    # supports maximization mode
    selector = LexicaseSelection(case_key="errors", case_max_best=True, random_state=123)

    a = make_individual("a", [10.0, 0.0])
    b = make_individual("b", [0.0, 10.0])
    c = make_individual("c", [5.0, 5.0])

    selected = selector.select(population_size=20, individuals=[a, b, c])

    # In maximization mode, only a or b should ever be selected.
    assert all(ind.phenotype in {"a", "b"} for ind in selected)
    assert all(ind.phenotype != "c" for ind in selected)


def test_lexicase_raises_when_case_data_missing():
    # raises when required case data is missing
    selector = LexicaseSelection(case_key="errors", case_max_best=False, random_state=1)

    ind = Individual(phenotype="a")
    ind.fitness = [0.0]

    with pytest.raises(ValueError, match="errors|case"):
        selector.select(population_size=1, individuals=[ind])



def test_epsilon_lexicase_with_zero_epsilon_matches_standard_behavior():
    # behaves like standard lexicase when epsilon is zero
    lex = LexicaseSelection(case_key="errors", case_max_best=False, random_state=123)
    eps = EpsilonLexicaseSelection(
        case_key="errors",
        epsilon=0.0,
        case_max_best=False,
        random_state=123,
    )

    individuals = [
        make_individual("a", [0.0, 10.0]),
        make_individual("b", [1.0, 0.0]),
        make_individual("c", [2.0, 2.0]),
    ]

    selected_lex = [ind.phenotype for ind in lex.select(30, individuals)]
    selected_eps = [ind.phenotype for ind in eps.select(30, individuals)]

    assert selected_lex == selected_eps


def test_epsilon_lexicase_allows_near_best_individuals():
    # epsilon allows near-best individuals to survive
    selector = EpsilonLexicaseSelection(
        case_key="errors",
        epsilon=0.2,
        case_max_best=False,
        random_state=123,
    )

    a = make_individual("a", [0.0, 10.0])
    b = make_individual("b", [0.1, 0.0])   # within epsilon on first case
    c = make_individual("c", [1.0, 1.0])   # outside epsilon

    selected = selector.select(population_size=30, individuals=[a, b, c])

    assert all(ind.phenotype in {"a", "b"} for ind in selected)
    assert all(ind.phenotype != "c" for ind in selected)


def test_epsilon_lexicase_supports_maximization():
    # supports maximization with epsilon
    selector = EpsilonLexicaseSelection(
        case_key="errors",
        epsilon=0.2,
        case_max_best=True,
        random_state=123,
    )

    a = make_individual("a", [10.0, 0.0])
    b = make_individual("b", [9.9, 10.0])   # within epsilon of case-0 best
    c = make_individual("c", [5.0, 5.0])

    selected = selector.select(population_size=30, individuals=[a, b, c])

    assert all(ind.phenotype in {"a", "b"} for ind in selected)
    assert all(ind.phenotype != "c" for ind in selected)


def test_epsilon_lexicase_raises_when_case_key_missing():
    # raises when required case key is missing
    selector = EpsilonLexicaseSelection(
        case_key="errors",
        epsilon=0.1,
        case_max_best=False,
        random_state=1,
    )

    ind = Individual(phenotype="a")
    ind.fitness = [0.0]
    ind.set_meta(Individual.CASE_DATA_META_KEY, {"other_key": [1.0, 2.0]})

    with pytest.raises(ValueError, match="errors|case key|Missing"):
        selector.select(population_size=1, individuals=[ind])