import pytest

from finchge.grammar.grammar import Grammar


@pytest.fixture
def sample_grammar():
    grammar_str = """
    <string> ::= <letter> | <letter> <string>
    <letter> ::= _ | [a-z]
    <number> ::= _ | 10..50 step 5
    """
    return Grammar(grammar_str)


@pytest.fixture
def sample_grammar_dt():
    grammar_str = """
<dt_params> ::= {
   "criterion" : <criterion> ,
   "max_depth" : <max_depth> ,
    "min_samples_leaf" : <min_samples_leaf> ,
    "min_samples_split" : <min_samples_split> ,
    "max_leaf_nodes" : <max_leaf_nodes>
}
<criterion> ::= "gini" | "entropy" | "log_loss"
<max_depth> ::= 10..70 step 10
<min_samples_leaf> ::= [1-3]
<min_samples_split> ::= [2-5]
<max_leaf_nodes> ::= 10..70 step 10
"""
    return Grammar(grammar_str)


def test_terminals_count(sample_grammar):
    """Test that terminals are counted correctly"""
    terminals = sample_grammar.terminals

    # Count terminals
    assert len(terminals) == 36

    # r
    assert "25" in terminals
    assert any("_" in str(t) for t in terminals)


def test_non_terminals_count(sample_grammar):
    """Test non-terminals count"""
    non_terminals = sample_grammar.non_terminals

    # Should have: {'<letter>', '<number>', '<string>'}
    assert len(non_terminals) == 3

    # Check each exists
    assert "<string>" in non_terminals
    assert "<letter>" in non_terminals
    assert "<number>" in non_terminals


def test_production_rules_count(sample_grammar):
    """Test number of production rules"""
    # Assuming grammar.rules or similar exists
    rules = sample_grammar.rules

    # Following 3 rules should exist
    """
    {'<string>': <string> ::= <letter> | <letter> <string>,
    '<letter>': <letter> ::= _ | [a-z],
    '<number>': <number> ::= _ | 10..50 step 5}
    """
    assert len(rules) == 3
    assert sample_grammar.start_rule == "<string>"


def test_grammar_validation(sample_grammar):
    """Test grammar is valid"""
    assert sample_grammar is not None


def test_grammar_analyse(sample_grammar_dt):
    sample_grammar_dt.analyze()

    # grammar should not be none
    assert sample_grammar_dt is not None
    # number of rules is 6
    assert len(sample_grammar_dt.rules) == 6
    # start rule
    assert sample_grammar_dt.start_rule == "<dt_params>"
    assert sample_grammar_dt.rules["<dt_params>"].min_path is not None
    # number of terminals and non terminals
    assert len(sample_grammar_dt.terminals) == 24
    assert len(sample_grammar_dt.non_terminals) == 6
    # Max arity
    assert sample_grammar_dt.max_arity > 0
    assert sample_grammar_dt.max_arity == 5
    # Can Terminate
    assert sample_grammar_dt.can_terminate
