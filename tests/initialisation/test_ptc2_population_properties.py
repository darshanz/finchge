import statistics

import pytest

from finchge.core.population import Population
from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import PTC2Initialiser, RampedPTC2Initialiser


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


def _tree(ind) -> TreeNode:
    assert ind.tree is not None
    return TreeNode.from_string(ind.tree)


def _tree_depth(ind) -> int:
    return _tree(ind).max_depth


def _tree_size(ind) -> int:
    return _tree(ind).size()


def test_ptc2_creates_full_population(tree_generator: TreeGenerator) -> None:
    init = PTC2Initialiser(target_size=12, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=30)

    assert len(population.individuals) == 30
    assert all(ind.tree is not None for ind in population.individuals)


def test_ptc2_population_is_tree_valid(tree_generator: TreeGenerator) -> None:
    init = PTC2Initialiser(target_size=12, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=30)

    for ind in population.individuals:
        tree = _tree(ind)
        phenotype = tree.to_phenotype()

        assert phenotype
        assert "<" not in phenotype


def test_ptc2_population_has_shape_diversity(tree_generator: TreeGenerator) -> None:
    init = PTC2Initialiser(target_size=12, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=30)

    sizes = [_tree_size(ind) for ind in population.individuals]
    depths = [_tree_depth(ind) for ind in population.individuals]

    assert len(set(sizes)) > 1
    assert len(set(depths)) > 1


def test_ptc2_population_has_phenotype_diversity(tree_generator: TreeGenerator) -> None:
    init = PTC2Initialiser(target_size=12, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=30)

    phenotypes = [_tree(ind).to_phenotype() for ind in population.individuals]
    unique_ratio = len(set(phenotypes)) / len(phenotypes)

    assert unique_ratio > 0.25


def test_ptc2_target_size_is_respected_on_average(
    tree_generator: TreeGenerator,
) -> None:
    init = PTC2Initialiser(target_size=12, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=50)

    sizes = [_tree_size(ind) for ind in population.individuals]
    mean_size = statistics.mean(sizes)

    # PTC2 is stochastic, so individual sizes should vary.
    # The mean should still be in the neighborhood of the requested target.
    assert mean_size > 4
    assert mean_size < 30


def test_ptc2_is_reproducible_for_same_seed(tree_generator: TreeGenerator) -> None:
    init1 = PTC2Initialiser(target_size=12, random_state=42)
    init1.set_tree_generator(tree_generator)

    init2 = PTC2Initialiser(target_size=12, random_state=42)
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 == trees2


def test_ptc2_changes_with_different_seed(tree_generator: TreeGenerator) -> None:
    init1 = PTC2Initialiser(target_size=12, random_state=42)
    init1.set_tree_generator(tree_generator)

    init2 = PTC2Initialiser(target_size=12, random_state=99)
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 != trees2


def test_ptc2d_respects_max_depth(tree_generator: TreeGenerator) -> None:
    init = PTC2Initialiser(target_size=12, max_depth=6, random_state=42)
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=30)

    depths = [_tree_depth(ind) for ind in population.individuals]

    assert max(depths) <= 6


def test_ramped_ptc2_creates_full_population(tree_generator: TreeGenerator) -> None:
    init = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=40,
        max_depth=None,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=40)

    assert len(population.individuals) == 40
    assert all(ind.tree is not None for ind in population.individuals)


def test_ramped_ptc2_produces_size_diversity(tree_generator: TreeGenerator) -> None:
    init = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=40,
        max_depth=None,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=40)

    sizes = [_tree_size(ind) for ind in population.individuals]

    assert len(set(sizes)) > 1
    assert min(sizes) > 0


def test_ramped_ptc2_produces_multiple_target_size_bins(
    tree_generator: TreeGenerator,
) -> None:
    init = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=40,
        max_depth=None,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=40)

    sizes = [_tree_size(ind) for ind in population.individuals]

    # We do not expect exact tree size == requested target size every time,
    # but ramped PTC2 should still generate a spread of structural sizes.
    assert len(set(sizes)) >= 3


def test_ramped_ptc2d_respects_max_depth(tree_generator: TreeGenerator) -> None:
    init = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=40,
        max_depth=6,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)

    population = Population(initialiser=init, population_size=40)

    depths = [_tree_depth(ind) for ind in population.individuals]
    assert max(depths) <= 6


def test_ramped_ptc2_is_reproducible_for_same_seed(
    tree_generator: TreeGenerator,
) -> None:
    init1 = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=20,
        max_depth=None,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=20,
        max_depth=None,
        random_state=42,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 == trees2


def test_ramped_ptc2_changes_with_different_seed(tree_generator: TreeGenerator) -> None:
    init1 = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=20,
        max_depth=None,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = RampedPTC2Initialiser(
        init_min_size=4,
        init_max_size=10,
        population_size=20,
        max_depth=None,
        random_state=99,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 != trees2
