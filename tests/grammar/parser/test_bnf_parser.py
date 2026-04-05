import pytest

from finchge.grammar.parser import BNFGrammarParser, consolidate_multiline_rules


class TestBNFGrammarParserBasic:
    """Test basic BNFGrammarParser functionality."""

    def test_parse_simple_grammar(self):
        """Test parsing a simple grammar."""

        simple_grammar = """
        <digit> ::= [0-9]
        <letter> ::= [a-z]
        <number> ::= 1..5
        """

        parser = BNFGrammarParser(simple_grammar)
        rules, rules_expanded, start, terminals, non_terminals = parser.parse()

        assert start == "<digit>"
        assert non_terminals == ["<digit>", "<letter>", "<number>"]
        assert "0" in terminals
        assert "9" in terminals
        assert "a" in terminals
        assert "z" in terminals
        assert "1" in terminals
        assert "5" in terminals

    def test_parse_complex_grammar(self):
        """
        Test parsing a complex grammar.
        normal grammar should parse correctly
        """

        complex_grammar = """
            <start> ::= <expr>
            <expr> ::= <term> | <expr> '+' <term>
            <term> ::= <factor> | <term> '*' <factor>
            <factor> ::= <number> | '(' <expr> ')'
            <number> ::= [0-9] | [0-9] [0-9]
            <digit_range> ::= 0..9
            <char_range> ::= 'a'..'z'
            <stepped> ::= 0..10 step 2
            """

        parser = BNFGrammarParser(complex_grammar)
        rules, rules_expanded, start, terminals, non_terminals = parser.parse()

        assert start == "<start>"
        assert "<expr>" in non_terminals
        assert "<term>" in non_terminals
        assert "<factor>" in non_terminals
        assert "<number>" in non_terminals

        # Check that ranges are expanded
        digit_rule = rules_expanded.get("<digit_range>")
        assert digit_rule is not None
        # Should have choices like ["0"], ["1"]

    def test_invalid_non_terminal(self):
        """
        Test that invalid non-terminals raise ValueError.
        This has invalid non terminal without angle brackets
        so raises error invalid non-terminal
        """
        invalid_grammar = """
        invalid ::= 'a'
        """
        parser = BNFGrammarParser(invalid_grammar)
        with pytest.raises(ValueError, match="Invalid non-terminal"):
            parser.parse()

    def test_unmatched_brackets(self):
        """
        Test that unmatched angle brackets doesn't affect parsing
        if bracket not closed it is treated as regular string
        """
        invalid_grammar = """
        <expr> ::= <unclosed
        """
        parser = BNFGrammarParser(invalid_grammar)
        _, _, _, terminals, _ = parser.parse()
        assert "<unclosed" in terminals

    def test_undefined_non_terminal(self):
        """
        Test that undefined non-terminals raise ValueError.
        <undefined> does not have any rule. need rule for each non-terminal
        """
        invalid_grammar = """
            <expr> ::= <undefined>
            """
        parser = BNFGrammarParser(invalid_grammar)
        with pytest.raises(ValueError, match="Undefined non-terminal"):
            parser.parse()


class TestBNFGrammarParserRangeExpansion:
    """Test range expansion in BNFGrammarParser."""

    def test_numeric_range_expansion(self):
        """Test that numeric ranges are expanded correctly."""
        grammar = """
        <range> ::= 1..3
        """
        parser = BNFGrammarParser(grammar)
        _, rules_expanded, _, terminals, _ = parser.parse()

        rule = rules_expanded["<range>"]
        # Should have one production with expanded range
        choices = rule.choices
        assert len(choices) == 3
        assert choices[0] == ["1"]
        assert "1" in terminals
        assert "2" in terminals
        assert "3" in terminals

    def test_char_range_expansion(self):
        """Test that character ranges are expanded correctly."""
        grammar = """
        <chars> ::= 'a'..'c'
        """
        parser = BNFGrammarParser(grammar)
        _, rules_expanded, _, terminals, _ = parser.parse()

        rule = rules_expanded["<chars>"]
        choices = rule.choices
        assert choices[0] == ["a"]
        assert choices[1] == ["b"]
        assert choices[2] == ["c"]
        assert "a" in terminals
        assert "b" in terminals
        assert "c" in terminals

    def test_char_class_expansion(self):
        """Test that character classes are expanded correctly."""
        grammar = """
        <digits> ::= [0-2]
        """
        parser = BNFGrammarParser(grammar)
        _, rules_expanded, _, terminals, _ = parser.parse()

        rule = rules_expanded["<digits>"]
        choices = rule.choices
        assert choices[0] == ["0"]
        assert choices[1] == ["1"]
        assert choices[2] == ["2"]
        assert "0" in terminals
        assert "1" in terminals
        assert "2" in terminals

    def test_mixed_range_types(self):
        """Test grammar with multiple range types."""
        grammar = """
        <test> ::= 1..3 | 'a'..'c' | [0-2]
        """
        parser = BNFGrammarParser(grammar)
        _, rules_expanded, _, terminals, _ = parser.parse()

        rule = rules_expanded["<test>"]

        # Should have 3 choices (one for each alternative)
        assert len(rule.choices) == 9

        # Check terminals include all expanded values
        expected_terminals = {"1", "2", "3", "a", "b", "c", "0", "1", "2"}
        for term in expected_terminals:
            assert term in terminals or term in {"1", "2"}  # duplicates OK

    def test_stepped_range_expansion(self):
        """
        Test stepped range expansion with line breaks.

        <stepped> ::= 0..6
        step 2

        should work same as :

        <stepped> ::= 0..6 step 2

        """
        grammar = """
        <stepped> ::= 0..6
        step 2
        """
        parser = BNFGrammarParser(grammar)
        _, rules_expanded, _, terminals, _ = parser.parse()

        rule = rules_expanded["<stepped>"]
        assert rule.choices[0] == ["0"]
        assert rule.choices[1] == ["2"]
        assert rule.choices[2] == ["4"]
        assert rule.choices[3] == ["6"]


class TestBNFGrammarParserMultiLine:
    """Test multi-line grammar parsing."""

    def test_multi_line_rule(self):
        """
        Test rule that spans multiple lines.
        a rule can be broken at any point

        """
        grammar = """
        <expr> ::= <term> |
        <expr> '+'
        <term>
        <term> ::= <factor> | <term> '*' <factor>
        <factor> ::= [2..4]
        """
        parser = BNFGrammarParser(grammar)
        rules, _, start, _, non_terminals = parser.parse()

        assert "<expr>" in non_terminals
        assert "<term>" in non_terminals
        assert "<factor>" in non_terminals

        expr_rule = rules["<expr>"]
        assert len(expr_rule.choices) == 2

        term_rule = rules["<term>"]
        assert len(term_rule.choices) == 2

    def test_comment_handling(self):
        """
        Test that comments are ignored.
        Comments on their own line or in-line comments at any position are acepted

        """
        grammar = """
        # This is a comment
        <expr> ::= <term>
        # own line comment
        <term> ::= 'x'  # Inline comment
        """
        parser = BNFGrammarParser(grammar)
        rules, _, _, _, non_terminals = parser.parse()

        assert "<expr>" in non_terminals
        assert "<term>" in non_terminals
        assert len(rules) == 2

    def test_consolidate_multiline_ranges_invalid(self):
        """Test consolidation of multiline rules with ranges."""
        grammar = """
            <digit> ::= [0-9]
            <stepped> ::= 1..10
            step 2
            """

        # step 2 in different line without prefix so should raise error
        try:
            lines = grammar.strip().split("\n")
            consolidate_multiline_rules(lines)
        except ValueError as e:
            assert "Missing '::='" in str(e)

    def test_line_break_after_choice(self):
        """Test consolidation of multiline rules with ranges.
        If line break happens with choice operators that's fine

        """
        separated_by_choice = """
                <e>  ::=  <e>+<e>|
                <e>-<e>|
                <c><c>.<c><c>
                <c>  ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
                """

        # Should pass when separated after choice
        lines = separated_by_choice.strip().split("\n")
        consolidated = consolidate_multiline_rules(lines)

        assert len(consolidated) == 2

    def test_complex_multiline_grammar(self):
        """Test complex grammar with multiple multiline rules."""
        grammar = """
        <expr> ::= <term>
            | <expr> '+' <term>  # choice operater after line break should work
            | <expr> '-' <term>
        <term> ::= <factor>
            | <term> '*' <factor> | # choice operater before line break should work
             <term> '/' <factor>
        <factor> ::= <number>
            | '(' <expr> ')'
        <number> ::= [0-9]
            | [0-9] | [0-9]
        """
        parser = BNFGrammarParser(grammar)
        rules, rules_expanded, start_rule, terminals, non_terminals = parser.parse()

        # Verify all rules are present
        expected_rules = ["<expr>", "<term>", "<factor>", "<number>"]
        assert non_terminals == expected_rules

        # Check expr has 3 alternatives
        expr_rule = rules["<expr>"]
        assert len(expr_rule.choices) == 3

        # Check term has 3 alternatives
        term_rule = rules["<term>"]
        assert len(term_rule.choices) == 3

        unquoted_terminals = set()
        for item in terminals:
            if item.startswith("'") and item.endswith("'"):
                unquoted_terminals.add(item[1:-1])  # Remove first and last character
            else:
                unquoted_terminals.add(item)

        # Check terminals
        assert "+" in unquoted_terminals
        assert "-" in unquoted_terminals
        assert "*" in unquoted_terminals
        assert "/" in unquoted_terminals
        assert "(" in unquoted_terminals
        assert ")" in unquoted_terminals

    def test_malformed_multiline_grammar(self):
        """test for wrong multiline grammars."""

        # Case 1 When rule separator on continuation line, this is incorrect format. so should result in error
        grammar1 = """
            <test> ::= 1..10
            ::= step 2  #  a new rule with empty LHS
            """

        parser1 = BNFGrammarParser(grammar1)

        # This should raise an error
        try:
            rules1, _, _, _, _ = parser1.parse()
        except ValueError as e:
            # expected behavior, should raise ValueError
            assert "Invalid non-terminal" in str(e)

    def test_real_world_multiline_example(self):
        """
        Test a realistic multiline grammar example.
        """
        grammar = """
        # Arithmetic expression grammar
        <expression> ::= <term>
                     | <expression> '+' <term>
                     | <expression> '-' <term>
        <term> ::= <factor>
               | <term> '*' <factor>
               | <term> '/' <factor>
        <factor> ::= <number>
                 | '(' <expression> ')'
                 | '+' <factor>
                 | '-' <factor>
        <number> ::= <digit>
                 | <number> <digit>
        <digit> ::= [0-9]
        # Range examples for testing
        <range_test> ::= 1..100 step 10
        <char_test> ::= 'A'..'Z' step 2
        """

        parser = BNFGrammarParser(grammar)
        rules, rules_expanded, start_rule, terminals, non_terminals = parser.parse()

        # Verify parsing
        assert start_rule == "<expression>"
        assert len(non_terminals) == 7

        # Check range_test expansion
        range_rule = rules_expanded["<range_test>"]
        range_production = [
            int(item) for sublist in range_rule.choices for item in sublist
        ]

        # Should have: 1, 11, 21, ..., 91
        assert 1 in range_production
        assert 11 in range_production
        assert 91 in range_production

        # Check char_test expansion
        char_rule = rules_expanded["<char_test>"]
        char_production = [
            str(item) for sublist in char_rule.choices for item in sublist
        ]
        # Should have: A, C, E, ..., Y
        assert "A" in char_production
        assert "C" in char_production
        assert "Y" in char_production
