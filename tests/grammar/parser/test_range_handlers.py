import re

import pytest

from finchge.grammar.range_handlers import (
    ArraySliceHandler,
    ArraySliceRangeHandler,
    CharClassHandler,
    CharRangeHandler,
    NumericRangeHandler,
    RangeHandler,
)


@pytest.fixture
def simple_grammar():
    """A simple grammar for basic testing."""
    return """
<digit> ::= [0-9]
<letter> ::= [a-z]
<number> ::= 1..5
"""


@pytest.fixture
def complex_grammar():
    """A complex grammar with multiple range types."""
    return """
<start> ::= <expr>
<expr> ::= <term> | <expr> '+' <term>
<term> ::= <factor> | <term> '*' <factor>
<factor> ::= <number> | '(' <expr> ')'
<number> ::= [0-9] | [0-9] [0-9]
<digit_range> ::= 0..9
<char_range> ::= 'a'..'z'
<stepped> ::= 0..10 step 2
"""


@pytest.fixture
def negative_range_grammar():
    """Grammar with negative ranges."""
    return """
    <negative> ::= -5..5
    <stepped_negative> ::= -10..0 step 2
    <descending> ::= 10..1 step -1
    """


@pytest.fixture
def numeric_range_handler():
    return NumericRangeHandler()


@pytest.fixture
def char_range_handler():
    return CharRangeHandler()


@pytest.fixture
def char_class_handler():
    return CharClassHandler()


class TestNumericRangeHandler:
    """Test NumericRangeHandler functionality."""

    def test_can_handle_numeric_range(self, numeric_range_handler):
        """Test that handler recognizes numeric ranges."""
        assert numeric_range_handler.can_handle("10..50") is True
        assert numeric_range_handler.can_handle("10..50 step 5") is True
        assert numeric_range_handler.can_handle("-5..5") is True
        assert numeric_range_handler.can_handle("-10..0 step 2") is True

    def test_cannot_handle_non_ranges(self, numeric_range_handler):
        """Test that handler rejects non-range tokens."""
        assert numeric_range_handler.can_handle("[0-9]") is False
        assert numeric_range_handler.can_handle("'a'..'z'") is False
        assert numeric_range_handler.can_handle("<nonterm>") is False
        assert numeric_range_handler.can_handle("plaintext") is False

    def test_expand_simple_range(self, numeric_range_handler):
        """Test expansion of simple numeric range."""
        result = numeric_range_handler.expand("1..5")
        assert result == ["1", "|", "2", "|", "3", "|", "4", "|", "5"]

    def test_expand_stepped_range(self, numeric_range_handler):
        """Test expansion of stepped numeric range."""
        result = numeric_range_handler.expand("1..10 step 2")
        assert result == ["1", "|", "3", "|", "5", "|", "7", "|", "9"]

    def test_expand_negative_range(self, numeric_range_handler):
        """Test expansion of negative range."""
        result = numeric_range_handler.expand("-2..2")
        assert result == ["-2", "|", "-1", "|", "0", "|", "1", "|", "2"]

    def test_expand_descending_range(self, numeric_range_handler):
        """Test expansion of descending range."""
        result = numeric_range_handler.expand("5..1")
        assert result == ["5", "|", "4", "|", "3", "|", "2", "|", "1"]

    def test_expand_negative_step(self, numeric_range_handler):
        """Test expansion with negative step."""
        result = numeric_range_handler.expand("5..1 step -1")
        assert result == ["5", "|", "4", "|", "3", "|", "2", "|", "1"]

    def test_expand_single_value(self, numeric_range_handler):
        """Test expansion of range with single value."""
        result = numeric_range_handler.expand("5..5")
        assert result == ["5"]


class TestCharRangeHandler:
    """Test CharRangeHandler functionality."""

    def test_can_handle_char_range(self, char_range_handler):
        """Test that handler recognizes character ranges."""
        assert char_range_handler.can_handle("'a'..'z'") is True
        assert char_range_handler.can_handle("'A'..'Z' step 2") is True
        assert char_range_handler.can_handle("'0'..'9'") is True

    def test_cannot_handle_non_char_ranges(self, char_range_handler):
        """Test that handler rejects non-character ranges."""
        assert char_range_handler.can_handle("10..50") is False
        assert char_range_handler.can_handle("[a-z]") is False

    def test_expand_char_range(self, char_range_handler):
        """Test expansion of character range."""
        result = char_range_handler.expand("'a'..'d'")
        assert result == ["a", "|", "b", "|", "c", "|", "d"]

    def test_expand_char_range_with_step(self, char_range_handler):
        """Test expansion of character range with step."""
        result = char_range_handler.expand("'a'..'f' step 2")
        assert result == ["a", "|", "c", "|", "e"]

    def test_expand_descending_char_range(self, char_range_handler):
        """Test expansion of descending character range."""
        result = char_range_handler.expand("'d'..'a'")
        assert result == ["d", "|", "c", "|", "b", "|", "a"]


class TestCharClassHandler:
    """Test CharClassHandler functionality."""

    def test_can_handle_char_class(self, char_class_handler):
        """Test that handler recognizes character classes."""
        assert char_class_handler.can_handle("[a-z]") is True
        assert char_class_handler.can_handle("[A-Za-z0-9_]") is True
        assert char_class_handler.can_handle("[0-9]") is True

    def test_cannot_handle_non_char_classes(self, char_class_handler):
        """Test that handler rejects non-character-class tokens."""
        assert char_class_handler.can_handle("'a'..'z'") is False
        assert char_class_handler.can_handle("10..50") is False

    def test_expand_simple_char_class(self, char_class_handler):
        """Test expansion of simple character class."""
        result = char_class_handler.expand("[a-d]")
        assert result == ["a", "|", "b", "|", "c", "|", "d"]

    def test_expand_multiple_ranges(self, char_class_handler):
        """Test expansion of character class with multiple ranges."""
        result = char_class_handler.expand("[a-cx-z]")
        # Note: This will deduplicate, so 'c' only appears once
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "x" in result
        assert "y" in result
        assert "z" in result
        assert result.count("|") == 5  # 6 items with 5 separators

    def test_expand_with_literals(self, char_class_handler):
        """Test expansion of character class with literal characters."""
        result = char_class_handler.expand("[abc]")
        assert result == ["a", "|", "b", "|", "c"]

    def test_expand_digits(self, char_class_handler):
        """Test expansion of digit range."""
        result = char_class_handler.expand("[0-9]")
        assert len(result) == 19  # 10 digits + 9 separators
        assert "0" in result
        assert "9" in result


class TestArraySliceHandler:
    """Test the ArraySliceHandler for fixed array slice terminals.
    Very important for symbolic regression.
    """

    def setup_method(self):
        self.handler = ArraySliceHandler()

    def test_can_handle_valid_array_slices(self):
        """
        Test that handler recognizes valid array slice patterns.
        - should handle all numbers
        - should handle large numbers
        - Space should not affect
        """
        valid_cases = [
            "x[:, 1]",  # Single digit
            "x[:, 10]",  # Multiple digits
            "x[:, 42]",  # Larger number
            "x[:, 0 ]",  # With space
            "x[:,  1]",  # Multiple spaces
            "x[ : , 2 ]",  # Spaces everywhere
        ]

        for case in valid_cases:
            assert self.handler.can_handle(case), f"Failed: {case}"

    def test_cannot_handle_invalid_patterns(self):
        """
        Test that handler rejects invalid patterns.
        - plain arrayslice handler does not handle ranges this is done by array slice range handler
        - alphabets can not be used to slice
        - if missing slicing index , it should fail
        - variable name is always x other letters for denoting dataset are not allowed
        - float index is not allowed
        - Only column slices are allowed row slices are not supported
        - should valid array slicing expression

        """
        invalid_cases = [
            "x[:, 0..4]",  # Range (should be handled by RangeHandler)
            "x[:, a]",  # Non-numeric
            "x[:,]",  # Missing index
            "x[:]",  # Wrong syntax
            "y[:, 0]",  # Wrong variable name
            "x[:, -1]",  # Negative index
            "x[:, 1.5]",  # Float index
            "x[0, :]",  # Row slice instead of column
            "x[:, 0] + 1",  # Expression, not just slice
        ]

        for case in invalid_cases:
            assert not self.handler.can_handle(case), f"Should reject: {case}"

    def test_expand_returns_single_item(self):
        """
        Test that expansion returns the slice as a single terminal.

        Array slice expansion is only for single slicers not for range

        """
        test_cases = [
            ("x[:, 0]", ["x[:, 0]"]),
            ("x[:, 42]", ["x[:, 42]"]),
            ("x[ : , 1 ]", ["x[ : , 1 ]"]),
        ]

        for input_slice, expected in test_cases:
            result = self.handler.expand(input_slice)
            assert result == expected, f"Failed for: {input_slice}"

    def test_pattern_property(self):
        """
        Test the regex pattern matches expected cases

        - Test to see if regex used is correct
            - see if the regex suports different cases.
        """
        pattern = self.handler.pattern

        # Should match valid cases
        assert re.fullmatch(pattern, "x[:, 0]")
        assert re.fullmatch(pattern, "x[:, 10]")
        assert re.fullmatch(pattern, "x[ : , 1 ]")

        # Should not match invalid cases
        assert not re.fullmatch(pattern, "x[:, 0..4]")
        assert not re.fullmatch(pattern, "x[:, a]")
        assert not re.fullmatch(pattern, "y[:, 0]")

    def test_priority(self):
        """Test that handler has appropriate priority."""
        assert self.handler.priority == 52  # It is set only after array slice range
        assert isinstance(self.handler, RangeHandler)


class TestArraySliceRangeHandler:
    """
    Test the ArraySliceRangeHandler for expanding array slice ranges.
    """

    def setup_method(self):
        self.handler = ArraySliceRangeHandler()

    def test_can_handle_array_slice_ranges(self):
        """Test that handler recognizes array slice range patterns."""
        valid_cases = [
            "x[:, 0..4]",  # Basic range
            "x[:, 0..10]",  # Larger range
            "x[:, 0..10 step 2]",  # With step
            "x[:, 5..15 step 3]",  # Different step
            "x[:, 1..1]",  # Single value range
            "x[ : , 0..4 ]",  # With spaces
            "x[:, 0..4 step 2 ]",  # Spaces with step
        ]

        for case in valid_cases:
            assert self.handler.can_handle(case), f"Failed: {case}"

    def test_cannot_handle_invalid_range_patterns(self):
        """Test that handler rejects invalid range patterns."""
        invalid_cases = [
            "x[:, 0]",  # Single slice (no range)
            "x[:, a..z]",  # Character range
            "x[:,]",  # Missing range
            "x[:, 0..]",  # Incomplete range
            "x[:, ..4]",  # Incomplete range
            "x[:, 0..4 step]",  # Missing step value
            "y[:, 0..4]",  # Wrong variable
        ]

        for case in invalid_cases:
            assert not self.handler.can_handle(case), f"Should reject: {case}"

    def test_expand_basic_range(self):
        """Test expansion of basic array slice range."""
        result = self.handler.expand("x[:, 0..4]")

        # Should expand to individual slices with | separators
        expected = [
            "x[:, 0]",
            "|",
            "x[:, 1]",
            "|",
            "x[:, 2]",
            "|",
            "x[:, 3]",
            "|",
            "x[:, 4]",
        ]

        assert result == expected

    def test_expand_with_step(self):
        """Test expansion of array slice range with step."""
        result = self.handler.expand("x[:, 0..10 step 2]")

        expected = [
            "x[:, 0]",
            "|",
            "x[:, 2]",
            "|",
            "x[:, 4]",
            "|",
            "x[:, 6]",
            "|",
            "x[:, 8]",
            "|",
            "x[:, 10]",
        ]

        assert result == expected

    def test_expand_single_value_range(self):
        """Test expansion of range with single value."""
        result = self.handler.expand("x[:, 5..5]")

        # Single value, no separators
        assert result == ["x[:, 5]"]

    def test_invalid_range_returns_original(self):
        """Test that invalid range expression returns original token."""
        # Mock a case where range extraction fails
        invalid_token = "x[:, invalid]"
        result = self.handler.expand(invalid_token)

        # Should return token as-is
        assert result == [invalid_token]

    def test_pattern_property(self):
        """Test the regex pattern matches range patterns."""
        pattern = self.handler.pattern

        # Should match range patterns
        assert re.fullmatch(pattern, "x[:, 0..4]")
        assert re.fullmatch(pattern, "x[:, 0..10 step 2]")
        assert re.fullmatch(pattern, "x[ : , 0..4 ]")

        # Should not match single slices
        assert not re.fullmatch(pattern, "x[:, 0]")
        assert not re.fullmatch(pattern, "x[:, a..z]")

    def test_priority_higher_than_numeric_handler(self):
        """Test that this handler has higher priority than numeric range handler."""
        # Array slice ranges should be handled before generic numeric ranges
        assert self.handler.priority == 53


class TestIntegrationWithParser:
    """Test integration of array slice handlers with the parser."""

    def test_parser_with_array_slice_handler(self):
        """Test that parser correctly handles array slices with the handler."""
        from finchge.grammar import Grammar

        # Grammar with array slices
        grammar = """
        <expr> ::= <var> | <const>
        <var> ::= x[:, 0] | x[:, 1] | x[:, 2]
        <const> ::= 1.0 | 2.0
        """

        grammar = Grammar(grammar)
        terminals = grammar.terminals

        # Check terminals include array slices
        assert "x[:, 0]" in terminals
        assert "x[:, 1]" in terminals
        assert "x[:, 2]" in terminals

    def test_parser_with_array_slice_range_handler(self):
        """Test that parser expands array slice ranges."""
        from finchge.grammar.parser import BNFGrammarParser

        # Grammar with array slice range
        grammar = """
<expr> ::= <var> | <const>
<var> ::= x[:, 0..2]
<const> ::= 1.0 | 2.0
"""

        parser = BNFGrammarParser(grammar)
        # Register the range handler
        parser.range_registry.register(ArraySliceRangeHandler())

        rules, expanded, start, terminals, non_terminals = parser.parse()

        # Check terminals include expanded array slices
        assert "x[:, 0]" in terminals
        assert "x[:, 1]" in terminals
        assert "x[:, 2]" in terminals

        # Check var rule was expanded to three alternatives
        var_rule = expanded["<var>"]
        assert len(var_rule.choices) == 3

    def test_handler_priority_resolution(self):
        """Test that array slice range handler takes priority over single slice handler."""
        from finchge.grammar.parser import RangeHandlerRegistry

        registry = RangeHandlerRegistry()

        # Register both handlers
        single_handler = ArraySliceHandler()
        range_handler = ArraySliceRangeHandler()

        registry.register(single_handler)
        registry.register(range_handler)

        # Range handler should have higher priority
        assert range_handler.priority > single_handler.priority


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_array_slice_with_large_numbers(self):
        """Test array slices with large column indices."""
        handler = ArraySliceHandler()

        # Should handle large numbers
        assert handler.can_handle("x[:, 100]")
        assert handler.can_handle("x[:, 999]")

        result = handler.expand("x[:, 100]")
        assert result == ["x[:, 100]"]

    def test_array_slice_range_with_large_step(self):
        """Test range expansion with step larger than range."""
        handler = ArraySliceRangeHandler()

        result = handler.expand("x[:, 0..5 step 10]")
        # Step 10 from 0 to 5: only 0 is included
        assert result == ["x[:, 0]"]

    def test_array_slice_with_different_brackets(self):
        """Test alternative bracket styles
        ONly square brackets are supported
        """
        handler = ArraySliceHandler()

        # Current implementation expects exactly this syntax
        assert not handler.can_handle("x(: , 0)")  # Parentheses
        assert not handler.can_handle("x{:, 0}")  # Curly braces

    def test_multi_dimensional_slices(self):
        """Test that multi-dimensional slices are not supported."""
        handler = ArraySliceHandler()

        # 2D slices should fail
        assert not handler.can_handle("x[0:10, 0]")
        assert not handler.can_handle("x[:, 0:10]")
        assert not handler.can_handle("x[0, :]")
