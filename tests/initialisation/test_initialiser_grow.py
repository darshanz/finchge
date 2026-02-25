import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import GrowTreeInitialiser


@pytest.fixture
def grammar():
    gramar_str = """
    <expr> ::= <expr><op><expr> | <var>
    <op> ::= + | -
    <var> ::= x | y
    """
    return Grammar(grammar_str=gramar_str)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def test_grow_initialiser_creates_individual(tree_generator):
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=42)
    init.set_tree_generator(tree_generator)

    ind = init.initialise()

    assert ind is not None
    assert ind.tree is not None


def test_tree_contains_valid_symbols(tree_generator, grammar):
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=42)
    init.set_tree_generator(tree_generator)

    ind = init.initialise()
    tree = TreeNode.from_string(ind.tree)

    allowed = set(grammar.non_terminals) | set(grammar.terminals)

    for node in tree.iter_nodes():
        assert node.symbol in allowed


def test_tree_respects_max_depth(tree_generator):
    max_depth = 12

    init = GrowTreeInitialiser(
        init_min_depth=6, init_max_depth=max_depth, random_state=42
    )
    init.set_tree_generator(tree_generator)

    ind = init.initialise()
    tree = TreeNode.from_string(ind.tree)

    assert tree.max_depth <= max_depth


def test_grow_initialiser_deterministic(tree_generator):
    init1 = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)
    init1.set_tree_generator(tree_generator)
    init2 = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)
    init2.set_tree_generator(tree_generator)

    ind1 = init1.initialise()
    ind2 = init2.initialise()

    assert ind1.tree == ind2.tree


def test_grow_initialiser_produces_variety(tree_generator):
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)
    init.set_tree_generator(tree_generator)

    trees = {init.initialise().tree for _ in range(5)}

    assert len(trees) > 1


def test_grow_initialiser_requires_tree_generator():
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)

    with pytest.raises(RuntimeError):
        init.initialise()


def test_tree_has_no_nonterminal_leaves(tree_generator, grammar):
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)
    init.set_tree_generator(tree_generator)

    ind = init.initialise()
    tree = TreeNode.from_string(ind.tree)

    for node in tree.iter_nodes():
        if not node.children:
            assert node.symbol not in grammar.non_terminals


def test_grow_initialiser_depth_distribution(tree_generator):
    init = GrowTreeInitialiser(init_min_depth=6, init_max_depth=12, random_state=123)
    init.set_tree_generator(tree_generator)

    depths = []

    for _ in range(20):
        tree = TreeNode.from_string(init.initialise().tree)
        depths.append(tree.max_depth)

    assert len(set(depths)) > 1
