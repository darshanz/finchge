import pytest

from finchge.core.population import Population
from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import PIGrowInitialiser


@pytest.fixture
def grammar() -> Grammar:
    grammar_str = """
    <expr> ::= <expr> <op> <expr>
             | ( <expr> )
             | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y | 1 | 2
    """
    gr = Grammar(grammar_str=grammar_str)
    gr.analyze()
    return gr


@pytest.fixture
def tree_generator(grammar: Grammar) -> TreeGenerator:
    return TreeGenerator(grammar=grammar, max_tree_depth=20)


@pytest.fixture
def pigrow_initialiser(tree_generator: TreeGenerator) -> PIGrowInitialiser:
    init = PIGrowInitialiser(
        init_max_depth=8,
        population_size=40,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)
    return init


def _tree(ind) -> TreeNode:
    assert ind.tree is not None
    return TreeNode.from_string(ind.tree)


def _tree_depth(ind) -> int:
    return _tree(ind).max_depth


def _tree_size(ind) -> int:
    return _tree(ind).size()


def test_pigrow_creates_full_population(pigrow_initialiser: PIGrowInitialiser) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    assert len(population.individuals) == 40
    assert all(ind is not None for ind in population.individuals)
    assert all(ind.tree is not None for ind in population.individuals)


def test_pigrow_population_is_tree_valid(pigrow_initialiser: PIGrowInitialiser) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    for ind in population.individuals:
        tree = _tree(ind)
        phenotype = tree.to_phenotype()

        assert phenotype
        assert "<" not in phenotype
        assert tree.max_depth <= 8


def test_pigrow_population_has_depth_diversity(
    pigrow_initialiser: PIGrowInitialiser,
) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    depths = [_tree_depth(ind) for ind in population.individuals]

    assert len(set(depths)) > 1
    assert min(depths) >= 3
    assert max(depths) <= 8


def test_pigrow_population_has_size_diversity(
    pigrow_initialiser: PIGrowInitialiser,
) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    sizes = [_tree_size(ind) for ind in population.individuals]

    assert len(set(sizes)) > 1
    assert min(sizes) > 0


def test_pigrow_population_has_phenotype_diversity(
    pigrow_initialiser: PIGrowInitialiser,
) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    phenotypes = [_tree(ind).to_phenotype() for ind in population.individuals]

    unique_ratio = len(set(phenotypes)) / len(phenotypes)
    assert unique_ratio > 0.25


def test_pigrow_reaches_requested_branch_depth(
    pigrow_initialiser: PIGrowInitialiser,
    grammar: Grammar,
) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=40)

    min_ramp = grammar.compute_min_ramp(
        population_size=40,
        max_init_depth=8,
    )
    assert min_ramp is not None

    expected_depths = set(range(min_ramp + 1, 8 + 1))
    observed_depths = {_tree_depth(ind) for ind in population.individuals}

    # PI-Grow should generate trees across the requested ramp range,
    # and at least some individuals should reach the intended target depths.
    assert observed_depths.intersection(expected_depths)
    assert len(observed_depths.intersection(expected_depths)) >= min(
        2, len(expected_depths)
    )


def test_pigrow_is_reproducible_for_same_seed(tree_generator: TreeGenerator) -> None:
    init1 = PIGrowInitialiser(
        init_max_depth=8,
        population_size=20,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = PIGrowInitialiser(
        init_max_depth=8,
        population_size=20,
        random_state=42,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 == trees2


def test_pigrow_changes_with_different_seed(tree_generator: TreeGenerator) -> None:
    init1 = PIGrowInitialiser(
        init_max_depth=8,
        population_size=20,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = PIGrowInitialiser(
        init_max_depth=8,
        population_size=20,
        random_state=99,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 != trees2


def test_pigrow_produces_more_than_one_unique_tree(
    pigrow_initialiser: PIGrowInitialiser,
) -> None:
    population = Population(initialiser=pigrow_initialiser, population_size=20)

    unique_trees = {ind.tree for ind in population.individuals}
    assert len(unique_trees) > 1
