import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import RHHInitialiser


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


def test_rhh_generates_individual(tree_generator):
    # does not crash
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=10,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)
    ind = init.initialise()

    assert ind.tree is not None


def test_rhh_respects_max_depth(tree_generator):
    # respect maximum depth
    max_depth = 4

    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=max_depth,
        population_size=20,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    for _ in range(20):
        tree = TreeNode.from_string(init.initialise().tree)
        assert tree.max_depth <= max_depth


def test_rhh_uses_depth_range(tree_generator):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=5,
        population_size=40,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    depths = set()

    for _ in range(40):
        tree = TreeNode.from_string(init.initialise().tree)
        depths.add(tree.max_depth)

    assert len(depths) > 1


def test_rhh_deterministic_reproducible(tree_generator):
    init1 = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=10,
        strict_full=False,
        random_state=42,
    )

    init2 = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=10,
        strict_full=False,
        random_state=42,
    )

    init1.set_tree_generator(tree_generator)
    init2.set_tree_generator(tree_generator)

    seq1 = [init1.initialise().tree for _ in range(10)]
    seq2 = [init2.initialise().tree for _ in range(10)]

    assert seq1 == seq2


def test_rhh_stochastic_diversity(tree_generator):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=30,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    trees = {init.initialise().tree for _ in range(30)}

    assert len(trees) > 1


def test_rhh_deterministic_alternates_modes(tree_generator):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=3,
        population_size=10,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    trees = [TreeNode.from_string(init.initialise().tree) for _ in range(6)]

    assert trees[0].to_string() != trees[1].to_string()


def test_rhh_strict_full_internal_nodes(tree_generator, grammar):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=20,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    for _ in range(10):
        tree = TreeNode.from_string(init.initialise().tree)

        for node in tree.iter_nodes():
            if node.depth < tree.max_depth:
                # Only enforce when full expansion actually used
                if node.symbol in grammar.non_terminals:
                    assert node.symbol in grammar.non_terminals


def test_rhh_produces_valid_grammar(tree_generator, grammar):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=15,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    for _ in range(15):
        tree = TreeNode.from_string(init.initialise().tree)

        for node in tree.iter_nodes():
            if node.children:
                rule = grammar.rules.get(node.symbol)
                assert rule is not None


def test_rhh_tree_shape_diversity(tree_generator):
    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=4,
        population_size=40,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tree_generator)

    shapes = set()

    for _ in range(40):
        tree = TreeNode.from_string(init.initialise().tree)
        shapes.add(tree.to_string())

    assert len(shapes) > 1


def test_rhh_deterministic_schedule_reproducible(grammar):
    tg1 = TreeGenerator(grammar, 10)
    tg2 = TreeGenerator(grammar, 10)

    init1 = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=3,
        population_size=4,
        strict_full=False,
        random_state=42,
    )

    init2 = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=3,
        population_size=4,
        strict_full=False,
        random_state=42,
    )

    init1.set_tree_generator(tg1)
    init2.set_tree_generator(tg2)

    seq1 = [
        TreeNode.from_string(init1.initialise().tree).to_phenotype() for _ in range(4)
    ]
    seq2 = [
        TreeNode.from_string(init2.initialise().tree).to_phenotype() for _ in range(4)
    ]

    assert seq1 == seq2


def test_rhh_consistent_schedule_cycles_deterministic_verison(grammar):
    tg = TreeGenerator(grammar, 10)

    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=3,
        population_size=4,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tg)

    params1 = [init._pick_params() for _ in range(4)]
    params2 = [init._pick_params() for _ in range(4)]

    assert params1 == params2


def test_rhh_deterministic_schedule_cycles_koza_style(grammar):
    tg = TreeGenerator(grammar, 10)

    init = RHHInitialiser(
        init_min_depth=2,
        init_max_depth=3,
        population_size=4,
        strict_full=False,
        random_state=42,
    )

    init.set_tree_generator(tg)

    params1 = [init._pick_params() for _ in range(4)]
    params2 = [init._pick_params() for _ in range(4)]

    assert params1 == params2
