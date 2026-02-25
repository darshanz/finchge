import pytest

from finchge.core.individual import Individual
from finchge.grammar.derivation_tree import TreeNode
from finchge.operators.crossover import SubtreeCrossover

tree_str = "6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{5:<var>{3:x_0}}}"

tree_str2 = "6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{6:<expr>,4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}},4:<op>,6:<expr>}}"


def make_individual(tree_):
    return Individual.from_tree(tree=TreeNode.from_string(tree_))


def test_cross_raises_error_if_tree_missing():
    #  subtree crossover raises a ValueError
    # if either parent does not have a tree defined.
    parent1 = Individual(tree=None)
    parent2 = make_individual(tree_=tree_str)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
    )

    with pytest.raises(ValueError):
        crossover.cross(parent1, parent2)


def test_cross_returns_two_offspring():
    # subtree crossover always returns two offspring.

    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1 is not None
    assert child2 is not None


def test_no_crossover_when_probability_zero():
    #   when crossover probability is zero,
    # offspring are clones of the parent trees.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=0.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1.tree == parent1.tree
    assert child2.tree == parent2.tree
    assert child1.tree is not parent1.tree
    assert child2.tree is not parent2.tree


def test_cross_swaps_subtrees_when_valid_symbols_exist():
    # subtree crossover swaps subtrees when parents
    # share valid non-terminal symbols.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=1,
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1.tree != parent1.tree or child2.tree != parent2.tree


def test_cross_does_not_modify_parent_trees():
    #  subtree crossover does not mutate parent trees.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    original_tree1 = parent1.tree
    original_tree2 = parent2.tree

    crossover = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=42,
    )

    crossover.cross(parent1, parent2)

    assert parent1.tree == original_tree1
    assert parent2.tree == original_tree2


def test_cross_is_deterministic_with_fixed_random_state():
    #  using the same random_state produces identical
    # offspring trees, ensuring reproducibility.
    parent1 = make_individual(tree_=tree_str)
    parent2 = make_individual(tree_=tree_str2)

    crossover1 = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=123,
    )

    crossover2 = SubtreeCrossover(
        crossover_proba=1.0,
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        random_state=123,
    )

    child1_a, child2_a = crossover1.cross(parent1, parent2)
    child1_b, child2_b = crossover2.cross(parent1, parent2)

    assert child1_a.tree == child1_b.tree
    assert child2_a.tree == child2_b.tree
