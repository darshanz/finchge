import pytest

from finchge.core.individual import Individual
from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.operators.mutation import SubtreeMutation

tree_str = "6:<expr>{6:<expr>{5:<var>{3:x_1}},4:<op>{1:+},6:<expr>{5:<var>{3:x_0}}}"


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
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def make_individual(tree_):
    return Individual.from_tree(tree=TreeNode.from_string(tree_))


def test_mutate_raises_error_if_tree_missing(tree_generator):
    # subtree mutation raises a ValueError
    # if the parent individual does not contain a tree.
    individual = Individual(tree=None)
    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=42,
    )
    with pytest.raises(ValueError):
        mutation.mutate(individual)


def test_mutate_returns_new_individual(tree_generator):
    # subtree mutation always returns a new individual.
    ind = make_individual(tree_=tree_str)

    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=42,
    )

    mutated = mutation.mutate(ind)
    assert mutated is not None
    assert mutated is not ind


def test_mutation_always_applies(tree_generator):
    """Regression test: SubtreeMutation no longer gates mutation behind a
    per-individual probability; every call must attempt mutation_events
    mutation events unconditionally."""
    ind = make_individual(tree_=tree_str)

    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=42,
    )

    child = mutation.mutate(ind)
    assert child.tree != ind.tree


def test_mutation_respects_mutation_events_count(tree_generator):
    # subtree mutation performs multiple mutation
    # events when mutation_events is greater than one.
    ind = make_individual(tree_=tree_str)

    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        mutation_events=2,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert child.tree is not None


def test_mutation_handles_no_valid_symbols_gracefully(tree_generator):
    # if no valid mutation symbols are found,
    # mutation stops without error and returns a cloned tree.
    ind = make_individual(tree_=tree_str)

    mutation = SubtreeMutation(
        non_terminals=["<unknown>"],  # does not exist in tree
        tree_generator=tree_generator,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert child.tree == ind.tree


def test_mutation_does_not_modify_parent_tree(tree_generator):
    # subtree mutation does not mutate
    # the parent individual's tree.
    ind = make_individual(tree_=tree_str)

    original_tree = ind.tree

    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=42,
    )
    mutation.mutate(ind)
    assert ind.tree == original_tree


def test_mutation_is_deterministic_with_fixed_random_state(tree_generator):
    # using the same random_state produces identical
    # mutated trees, ensuring reproducibility.
    ind = make_individual(tree_=tree_str)

    mutation1 = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=123,
    )

    mutation2 = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        random_state=123,
    )
    child1 = mutation1.mutate(ind)
    child2 = mutation2.mutate(ind)
    assert child1.tree == child2.tree


def test_mutation_never_exceeds_global_max_tree_depth(grammar):
    # A mutation point deep enough that a large replacement subtree would
    # exceed max_tree_depth must still respect the global cap.
    tree_generator = TreeGenerator(grammar=grammar, max_tree_depth=8)
    deep_tree = tree_generator._generate_depthfirst_tree(
        start_symbol="<expr>",
        max_depth=8,
        method="grow",
        rng=__import__("random").Random(0),
    )

    mutation = SubtreeMutation(
        non_terminals=["<expr>", "<op>", "<var>", "<const>"],
        tree_generator=tree_generator,
        mutation_events=5,
        random_state=42,
    )

    ind = Individual.from_tree(tree=deep_tree)
    for _ in range(20):  # repeat to exercise different random mutation points
        ind = mutation.mutate(ind)
        assert TreeNode.from_string(ind.tree).max_depth <= 8
