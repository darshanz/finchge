import re

import numpy as np

from finchge.grammar.range_handlers import (
    RangeHandler,
    SymbolicVariableRangeHandler,
    SymolicVariableHandler,
)
from finchge.symbolic import SymbolicExpression


def test_expression_evaluation():
    handler = SymbolicVariableRangeHandler()
    expanded = handler.expand("x[0..2]")
    symbols = [var for var in expanded if var != "|"]
    expr = SymbolicExpression("+".join(symbols))  # x0 + x1 + x2)
    X = np.array([[1, 2, 3], [4, 5, 6]])
    result = expr.eval(X)
    assert np.array_equal(result, [1 + 2 + 3, 4 + 5 + 6])


class TestSymbolicVariableHandler:
    """Test the SymolicVariableHandler for fixed variables in terminals.
    important for symbolic regression.
    """

    def setup_method(self):
        self.handler = SymolicVariableHandler()

    def test_can_handle_valid_variable(self):
        """
        Test that handler recognizes valid symbols.
        - should handle all numbers
        - should handle large numbers
        """
        valid_cases = [
            "x1",  # Single digit
            "x10",  # Multiple digits
            "x42",  # Larger number
        ]

        for case in valid_cases:
            assert self.handler.can_handle(case), f"Failed: {case}"

    def test_cannot_handle_invalid_patterns(self):
        """
        Test that handler rejects invalid patterns.
        - plain variable handler does not handle ranges this is done by variable range handler
        - alphabets can not be used in the column index
        - if missing column index , it should fail
        - variable name is always x other letters for denoting dataset are not allowed
        - float index is not allowed

        """
        invalid_cases = [
            "x[0..4]",  # Range (should be handled by RangeHandler)
            "xa",  # Non-numeric
            "x",  # Missing index
            "x[",  # Wrong syntax
            "y0",  # Wrong variable name
            "x-1",  # Negative index
            "x1.5",  # Float index
            "x0 + 1",  # Expression, not just variable
        ]

        for case in invalid_cases:
            assert not self.handler.can_handle(case), f"Should reject: {case}"

    def test_pattern_property(self):
        """
        Test the regex pattern matches expected cases

        - Test to see if regex used is correct
         - see if the regex suports different cases.
        """
        pattern = self.handler.pattern

        # Should match valid cases
        assert re.fullmatch(pattern, "x0")
        assert re.fullmatch(pattern, "x10")
        assert re.fullmatch(pattern, "x1")

        # Should not match invalid cases
        assert not re.fullmatch(pattern, "x[0..4]")
        assert not re.fullmatch(pattern, "xa")
        assert not re.fullmatch(pattern, "y0")

    def test_priority(self):
        """Test that handler has appropriate priority."""
        assert (
            self.handler.priority == 50
        )  # It is set only after SymbolicVariableRangeHandler
        assert isinstance(self.handler, RangeHandler)


class TestSymbolicVariableRangeHandler:
    """
    Test the SymbolicVariableRangeHandler for expanding variable ranges.
    """

    def setup_method(self):
        self.handler = SymbolicVariableRangeHandler()

    def test_can_handle_variablee_ranges(self):
        """Test that handler recognizes rymbolic rariable range  patterns."""
        valid_cases = [
            "x[0..4]",  # Basic range
            "x[0..10]",  # Larger range
            "x[0..10 step 2]",  # With step
            "x[5..15 step 3]",  # Different step
            "x[1..1]",  # Single value range
            "x[ 0..4 ]",  # With spaces
            "x[0..4 step 2 ]",  # Spaces with step
        ]

        for case in valid_cases:
            assert self.handler.can_handle(case), f"Failed: {case}"

    def test_cannot_handle_invalid_range_patterns(self):
        """Test that handler rejects invalid range patterns."""
        invalid_cases = [
            "x0",  # Single slice (no range)
            "x[a..z]",  # Character range
            "x[]",  # Missing range
            "x[0..]",  # Incomplete range
            "x[..4]",  # Incomplete range
            "x[0..4 step]",  # Missing step value
            "y[0..4]",  # Wrong variable
        ]

        for case in invalid_cases:
            assert not self.handler.can_handle(case), f"Should reject: {case}"

    def test_expand_basic_range(self):
        """Test expansion of basic variable range."""
        result = self.handler.expand("x[0..4]")

        # Should expand to individual slices with | separators
        expected = [
            "x0",
            "|",
            "x1",
            "|",
            "x2",
            "|",
            "x3",
            "|",
            "x4",
        ]

        assert result == expected

    def test_expand_with_step(self):
        """Test expansion of variable range with step."""
        result = self.handler.expand("x[0..10 step 2]")

        expected = [
            "x0",
            "|",
            "x2",
            "|",
            "x4",
            "|",
            "x6",
            "|",
            "x8",
            "|",
            "x10",
        ]

        assert result == expected

    def test_expand_single_value_range(self):
        """Test expansion of range with single value."""
        result = self.handler.expand("x[5..5]")

        # Single value, no separators
        assert result == ["x5"]

    def test_invalid_range_returns_original(self):
        """Test that invalid range expression returns original token."""
        # Mock a case where range extraction fails
        invalid_token = "x[invalid]"
        result = self.handler.expand(invalid_token)

        # Should return token as-is
        assert result == [invalid_token]

    def test_pattern_property(self):
        """Test the regex pattern matches range patterns."""
        pattern = self.handler.pattern

        # Should match range patterns
        assert re.fullmatch(pattern, "x[0..4]")
        assert re.fullmatch(pattern, "x[0..10 step 2]")
        assert re.fullmatch(pattern, "x[0..4 ]")

        # Should not match single slices
        assert not re.fullmatch(pattern, "x0")
        assert not re.fullmatch(pattern, "x[a..z]")

    def test_priority_higher_than_numeric_handler(self):
        """Test that this handler has higher priority than numeric range handler."""
        # Array slice ranges should be handled before generic numeric ranges
        assert self.handler.priority == 52


class TestIntegrationWithParser:
    """Test integration of array slice handlers with the parser."""

    def test_parser_with_variable_handler(self):
        """Test that parser correctly handles variables with the handler."""
        from finchge.grammar import Grammar

        # Grammar with array slices
        grammar = """
        <expr> ::= <var> | <const>
        <var> ::= x0 | x1 | x2
        <const> ::= 1.0 | 2.0
        """

        grammar = Grammar(grammar)
        terminals = grammar.terminals

        # Check terminals include array slices
        assert "x0" in terminals
        assert "x1" in terminals
        assert "x2" in terminals

    def test_parser_with_variable_range_handler(self):
        """Test that parser expands variable ranges."""
        from finchge.grammar.parser import BNFGrammarParser

        # Grammar with array slice range
        grammar = """
        <expr> ::= <var> | <const>
        <var> ::= x[0..2]
        <const> ::= 1.0 | 2.0
        """

        parser = BNFGrammarParser(grammar)
        # Register the range handler
        parser.range_registry.register(SymbolicVariableRangeHandler())

        rules, expanded, start, terminals, non_terminals = parser.parse()

        # Check terminals include expanded array slices
        assert "x0" in terminals
        assert "x1" in terminals
        assert "x2" in terminals

        # Check var rule was expanded to three alternatives
        var_rule = expanded["<var>"]
        assert len(var_rule.choices) == 3

    def test_handler_priority_resolution(self):
        """Test that range handler takes priority over single variable handler."""
        from finchge.grammar.parser import RangeHandlerRegistry

        registry = RangeHandlerRegistry()

        # Register both handlers
        single_handler = SymolicVariableHandler()
        range_handler = SymbolicVariableRangeHandler()

        registry.register(single_handler)
        registry.register(range_handler)

        # Range handler should have higher priority
        assert range_handler.priority > single_handler.priority


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_variable_with_large_numbers(self):
        """Test array slices with large column indices."""
        handler = SymolicVariableHandler()

        # Should handle large numbers
        assert handler.can_handle("x100")
        assert handler.can_handle("x999")

        result = handler.expand("x100")
        assert result == ["x100"]

    def test_variable_range_with_large_step(self):
        """Test range expansion with step larger than range."""
        handler = SymbolicVariableRangeHandler()

        result = handler.expand("x[0..5 step 10]")
        # Step 10 from 0 to 5: only 0 is included
        assert result == ["x0"]


def test_grammar_parsing_with_mixed_syntax():
    """Test that parser correctly handles mixed individual and range syntax."""
    from finchge.grammar import Grammar

    grammar_text = """
    <expr> ::= <var> <op> <var> | <const>
    <var> ::= x0 | x[1..3] | x4  # Mix of individual and range
    <op> ::= + | - | *
    <const> ::= 1.0 | 2.0 | 3.0
    """

    grammar = Grammar(grammar_text)

    # Verify terminals include all expanded variables
    terminals = grammar.terminals
    expected_terminals = {
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "+",
        "-",
        "*",
        "1.0",
        "2.0",
        "3.0",
    }

    for term in expected_terminals:
        assert term in terminals, f"Missing terminal: {term}"

    # Verify <var> rule was properly expanded
    var_rule = grammar.rules.get("<var>", [])
    assert len(var_rule.choices) == 5  # x0, x1, x2, x3, x4


def test_malformed_range_expressions():
    """Test handling of malformed range expressions."""
    handler = SymbolicVariableRangeHandler()

    malformed_cases = [
        ("x[0..-1]", []),  # Negative end
        ("x[-1..5]", []),  # Negative start
        ("x[0..4 step -1]", []),  # Negative step
        ("x[1.5..4]", []),  # Float in range
        ("x[0..]", []),  # Incomplete
        ("x[..4]", []),  # Incomplete
    ]

    for token, _ in malformed_cases:
        result = handler.expand(token)
        # Should either return the token as-is or empty list
        assert result == [token] or result == []


def test_whitespace_variations():
    """Test that handler is robust to whitespace variations."""
    handler = SymbolicVariableRangeHandler()

    whitespace_variations = [
        "x[0..4]",
        "x[0..4 ]",
        "x[ 0..4]",
        "x[ 0..4 ]",
        "x[0..4 step 2]",
        "x[0..4 step 2 ]",
        "x[0..4  step  2]",
    ]

    for token in whitespace_variations:
        assert handler.can_handle(token), f"Failed: {token}"
        result = handler.expand(token)
        # Should all produce the same expansion
        if "step" in token:
            assert "x0" in result and "x2" in result and "x4" in result
        else:
            assert all(f"x{i}" in result for i in range(5))


def test_backward_compatibility_layer():
    """Test a compatibility layer for old syntax."""

    def convert_old_to_new(expr: str) -> str:
        import re

        # Convert x[:, n] to xn
        expr = re.sub(r"x\[\s*:\s*,\s*(\d+)\s*\]", r"x\1", expr)
        # Convert x[:, m..n] to x[m..n]
        expr = re.sub(
            r"x\[\s*:\s*,\s*(\d+\.\.\d+(?:\s+step\s+\d+)?)\s*\]", r"x[\1]", expr
        )
        return expr

    test_cases = [
        ("x[:, 0]", "x0"),
        ("x[:, 1] + x[:, 2]", "x1 + x2"),
        ("sin(x[:, 0..2])", "sin(x[0..2])"),
        ("x[:, 0..4 step 2]", "x[0..4 step 2]"),
    ]

    for old, expected in test_cases:
        result = convert_old_to_new(old)
        assert result == expected, f"Failed: {old} -> {result}, expected {expected}"
