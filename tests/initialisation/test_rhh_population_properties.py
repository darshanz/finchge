import pytest

from finchge.core.population import Population
from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import RHHInitialiser


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
def rhh_initialiser(tree_generator: TreeGenerator) -> RHHInitialiser:
    init = RHHInitialiser(
        init_max_depth=8,
        population_size=40,
        strict_full=False,
        random_state=42,
    )
    init.set_tree_generator(tree_generator)
    return init


def _tree_depth(ind) -> int:
    assert ind.tree is not None
    return TreeNode.from_string(ind.tree).max_depth


def _tree_size(ind) -> int:
    assert ind.tree is not None
    return TreeNode.from_string(ind.tree).size()


def test_rhh_creates_full_population(rhh_initialiser: RHHInitialiser) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    assert len(population.individuals) == 40
    assert all(ind is not None for ind in population.individuals)
    assert all(ind.tree is not None for ind in population.individuals)


def test_rhh_population_is_tree_valid(
    grammar: Grammar, rhh_initialiser: RHHInitialiser
) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    for ind in population.individuals:
        assert ind.tree is not None
        tree = TreeNode.from_string(ind.tree)
        phenotype = tree.to_phenotype()

        assert phenotype
        assert "<" not in phenotype
        assert tree.max_depth <= 8


def test_rhh_population_has_depth_diversity(rhh_initialiser: RHHInitialiser) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    depths = [_tree_depth(ind) for ind in population.individuals]

    assert len(set(depths)) > 1
    assert min(depths) >= 3
    assert max(depths) <= 8


def test_rhh_population_has_size_diversity(rhh_initialiser: RHHInitialiser) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    sizes = [_tree_size(ind) for ind in population.individuals]

    assert len(set(sizes)) > 1
    assert min(sizes) > 0


def test_rhh_population_has_phenotype_diversity(
    rhh_initialiser: RHHInitialiser,
) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    phenotypes = []
    for ind in population.individuals:
        assert ind.tree is not None
        phenotypes.append(TreeNode.from_string(ind.tree).to_phenotype())

    unique_ratio = len(set(phenotypes)) / len(phenotypes)
    assert unique_ratio > 0.25


def test_rhh_is_reproducible_for_same_seed(tree_generator: TreeGenerator) -> None:
    init1 = RHHInitialiser(
        init_max_depth=8,
        population_size=20,
        strict_full=False,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = RHHInitialiser(
        init_max_depth=8,
        population_size=20,
        strict_full=False,
        random_state=42,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 == trees2


def test_rhh_changes_with_different_seed(tree_generator: TreeGenerator) -> None:
    init1 = RHHInitialiser(
        init_max_depth=8,
        population_size=20,
        strict_full=False,
        random_state=42,
    )
    init1.set_tree_generator(tree_generator)

    init2 = RHHInitialiser(
        init_max_depth=8,
        population_size=20,
        strict_full=False,
        random_state=99,
    )
    init2.set_tree_generator(tree_generator)

    pop1 = Population(initialiser=init1, population_size=20)
    pop2 = Population(initialiser=init2, population_size=20)

    trees1 = [ind.tree for ind in pop1.individuals]
    trees2 = [ind.tree for ind in pop2.individuals]

    assert trees1 != trees2


def test_rhh_covers_multiple_ramp_depths(
    grammar: Grammar, rhh_initialiser: RHHInitialiser
) -> None:
    population = Population(initialiser=rhh_initialiser, population_size=40)

    observed_depths = {_tree_depth(ind) for ind in population.individuals}

    min_ramp = grammar.compute_min_ramp(
        population_size=40,
        max_init_depth=8,
    )
    assert min_ramp is not None

    expected_depths = set(range(min_ramp + 1, 8 + 1))

    # RHH should cover at least part of the grammar-derived ramp range.
    # If population is large enough, it should usually cover all of it.
    assert observed_depths.intersection(expected_depths)
    assert len(observed_depths.intersection(expected_depths)) >= min(
        2, len(expected_depths)
    )
