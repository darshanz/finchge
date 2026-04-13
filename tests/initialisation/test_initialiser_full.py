import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import FullTreeInitialiser


@pytest.fixture
def grammar():
    gramar_str = """
    <expr> ::= <expr> <op> <expr>
         | ( <expr> )
         | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y | 1 | 2
    """
    return Grammar(grammar_str=gramar_str)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def test_full_initialiser_creates_individual(tree_generator):
    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=20, strict_full=False, random_state=42
    )
    init.set_tree_generator(tree_generator)

    ind = init.initialise()

    assert ind is not None
    assert ind.tree is not None


def test_full_tree_respects_max_depth_soft(tree_generator):
    """
    Tree May not reach Maximum Depth when strict_full is false.
    But should follow: tree.max_depth <= max_depth
    """
    max_depth = 10

    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=max_depth, strict_full=False, random_state=42
    )

    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.max_depth <= max_depth


def test_full_internal_nodes_are_nonterminals(tree_generator, grammar):
    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=42
    )
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    for node in tree.iter_nodes():
        if node.children:
            assert node.symbol in grammar.non_terminals


def test_full_terminals_only_at_leaves(tree_generator, grammar):
    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=42
    )
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    for node in tree.iter_nodes():
        if node.symbol in grammar.terminals:
            assert not node.children


def test_full_initialiser_deterministic(tree_generator):
    init1 = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=123
    )
    init1.set_tree_generator(tree_generator)

    init2 = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=123
    )
    init2.set_tree_generator(tree_generator)

    ind1 = init1.initialise()
    ind2 = init2.initialise()

    assert ind1.tree == ind2.tree


def test_full_requires_tree_generator():
    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=42
    )

    with pytest.raises(RuntimeError):
        init.initialise()


def test_full_tree_symbols_valid(tree_generator, grammar):
    init = FullTreeInitialiser(
        init_min_depth=3, init_max_depth=4, strict_full=False, random_state=42
    )
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    allowed = set(grammar.terminals) | set(grammar.non_terminals)

    for node in tree.iter_nodes():
        assert node.symbol in allowed
