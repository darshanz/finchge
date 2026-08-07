import pytest

from finchge import Grammar
from finchge.core.individual import Individual
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.operators.crossover import SubtreeCrossover

tree_str = "6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{5:<var>{3:x_0}}}"

tree_str2 = "6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>,4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}}"


@pytest.fixture
def grammar():
    grammar_str = """<expr> ::= <expr> <op> <expr> | <var> | <const>

    <op> ::= + | -

    <var> ::= x_0 | x_1

    <const> ::= 1.0 | 2.0 | 3.0
    """
    return Grammar(grammar_str=grammar_str)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=20)


def make_individual(tree_):
    return Individual.from_tree(tree=TreeNode.from_string(tree_))


def test_cross_raises_error_if_tree_missing(tree_generator):
    #  subtree crossover raises a ValueError
    # if either parent does not have a tree defined.
    parent1 = Individual(tree=None)
    parent2 = make_individual(tree_=tree_str)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
        tree_generator=tree_generator,
    )

    with pytest.raises(ValueError):
        crossover.cross(parent1, parent2)


def test_cross_returns_two_offspring(tree_generator):
    # subtree crossover always returns two offspring.

    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
        tree_generator=tree_generator,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1 is not None
    assert child2 is not None


def test_no_crossover_when_probability_zero(tree_generator):
    #   when crossover probability is zero,
    # offspring are clones of the parent trees.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=0.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
        tree_generator=tree_generator,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1.tree == parent1.tree
    assert child2.tree == parent2.tree
    assert child1.tree is not parent1.tree
    assert child2.tree is not parent2.tree


def test_cross_swaps_subtrees_when_valid_symbols_exist(tree_generator):
    # subtree crossover swaps subtrees when parents
    # share valid non-terminal symbols.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=1,
        tree_generator=tree_generator,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1.tree != parent1.tree or child2.tree != parent2.tree


def test_cross_does_not_modify_parent_trees(tree_generator):
    #  subtree crossover does not mutate parent trees.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    original_tree1 = parent1.tree
    original_tree2 = parent2.tree

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
        tree_generator=tree_generator,
    )

    crossover.cross(parent1, parent2)

    assert parent1.tree == original_tree1
    assert parent2.tree == original_tree2


def test_cross_is_deterministic_with_fixed_random_state(tree_generator):
    #  using the same random_state produces identical
    # offspring trees, ensuring reproducibility.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover1 = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=123,
        tree_generator=tree_generator,
    )

    crossover2 = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=123,
        tree_generator=tree_generator,
    )

    child1_a, child2_a = crossover1.cross(parent1, parent2)
    child1_b, child2_b = crossover2.cross(parent1, parent2)

    assert child1_a.tree == child1_b.tree
    assert child2_a.tree == child2_b.tree


def test_cross_never_exceeds_max_tree_depth(grammar):
    # Parents generated at close to the cap
    # repeated crossover attempts must never produce a child deeper than max_tree_depth,
    # even though a naive swap could exceed it.
    import random

    tg = TreeGenerator(grammar=grammar, max_tree_depth=8)
    rng = random.Random(0)
    t1 = tg.generate_tree_grow(start_symbol="<expr>", max_depth=8, rng=rng)
    t2 = tg.generate_tree_grow(start_symbol="<expr>", max_depth=8, rng=rng)
    parent1 = Individual.from_tree(t1)
    parent2 = Individual.from_tree(t2)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tg,
        random_state=42,
    )

    for _ in range(20):  # repeat to exercise different random crossover points
        child1, child2 = crossover.cross(parent1, parent2)
        assert TreeNode.from_string(child1.tree).max_depth <= 8
        assert TreeNode.from_string(child2.tree).max_depth <= 8
