import re

from finchge.grammar import BNFGrammarParser
from finchge.grammar.range_handlers import RangeHandler


class TestIntegration:
    """Integration tests for the complete parsing system."""

    def test_custom_handler_integration(self):
        """Test adding a custom range handler."""
        from finchge.grammar.range_handlers import RangeHandler

        class CustomRangeHandler(RangeHandler):
            """Handles negative numbers: -10..10 or -10..10 step 2"""

            def can_handle(self, token: str) -> bool:
                import re

                return bool(re.fullmatch(r"-?\d+\.\.-?\d+(?:\s+step\s+-?\d+)?", token))

            def expand(self, token: str):
                import re

                match = re.match(r"(-?\d+)\.\.(-?\d+)(?:\s+step\s+(-?\d+))?", token)
                if not match:
                    return []

                start, end, step = match.groups()
                start, end = int(start), int(end)
                step = int(step) if step else 1

                if start <= end:
                    items = list(range(start, end + 1, step))
                else:
                    items = list(range(start, end - 1, -step))

                expanded = []
                for i, item in enumerate(items):
                    expanded.append(str(item))
                    if i < len(items) - 1:
                        expanded.append("|")
                return expanded

            @property
            def pattern(self) -> str:
                return r"-?\d+\.\.-?\d+(?:\s+step\s+-?\d+)?"

        # Test grammar with hex ranges
        grammar = """
        <stepped_negative> ::= -10..0 step 2
        """

        parser = BNFGrammarParser(grammar)
        parser.range_registry.register(CustomRangeHandler())

        rules, rules_expanded, _, terminals, _ = parser.parse()

        neg_rule = rules_expanded["<stepped_negative>"]
        assert neg_rule.choices[0] == ["-10"]
        assert neg_rule.choices[1] == ["-8"]
        assert neg_rule.choices[2] == ["-6"]
        assert neg_rule.choices[3] == ["-4"]
        assert neg_rule.choices[4] == ["-2"]
        assert neg_rule.choices[5] == ["0"]

        assert "-4" in terminals
        assert "-6" in terminals
        assert "-8" in terminals
        assert "-2" in terminals

    def test_devanagari_range_handler(self):
        """Test Devanagari alphabet range handler with high priority."""

        class DevanagariRangeHandler(RangeHandler):
            """Handles Devanagari character ranges"""

            def __init__(self) -> None:
                super().__init__(priority=15)  # Higher than generic char ranges

            def can_handle(self, token: str) -> bool:
                # Check if it's a quoted Devanagari range: 'अ'..'ह'
                if not re.fullmatch(r"'[^']+'\.\.'[^']+'(?:\s+step\s+\d+)?", token):
                    return False

                # Extract characters and check if they're Devanagari
                match = re.match(r"'([^']*)'\.\.'([^']*)'", token)
                if not match:
                    return False

                start_char, end_char = match.groups()
                # Check if both are single Devanagari characters
                if len(start_char) != 1 or len(end_char) != 1:
                    return False

                # Devanagari Unicode range is U+0900 to U+097F
                # Reference : https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
                return (
                    "\u0900" <= start_char <= "\u097f"
                    and "\u0900" <= end_char <= "\u097f"
                )

            def expand(self, token: str) -> list[str]:
                match = re.match(r"'([^']*)'\.\.'([^']*)'(?:\s+step\s+(\d+))?", token)
                start_char, end_char, step = match.groups()
                step = int(step) if step else 1

                start_code, end_code = ord(start_char), ord(end_char)

                if start_code <= end_code:
                    chars = [chr(c) for c in range(start_code, end_code + 1, step)]
                else:
                    chars = [chr(c) for c in range(start_code, end_code - 1, -step)]

                # Add | separators
                expanded = []
                for i, char in enumerate(chars):
                    expanded.append(char)
                    if i < len(chars) - 1:
                        expanded.append("|")
                return expanded

            @property
            def pattern(self) -> str:
                # Same pattern as CharRangeHandler but will only match Devanagari in can_handle()
                return r"'[^']+'\.\.'[^']+'(?:\s+step\s+\d+)?"

        # Test grammar with mixed ranges
        grammar = """
        <devanagari> ::= 'अ'..'आ'
        <generic> ::= 'a'..'z'
        <mixed> ::= 'अ'..'आ' | 'a'..'z'
        """

        parser = BNFGrammarParser(grammar)
        parser.range_registry.register(DevanagariRangeHandler())

        rules, rules_expanded, _, terminals, _ = parser.parse()

        # Check that Devanagari handler processed 'अ'..'आ'
        dev_rule = rules_expanded["<devanagari>"]
        # Should have: ['अ', '|', 'आ'] or ['अ', '|', 'अ', '|', 'आ'] depending on what's between
        assert any("अ" in prod for prod in dev_rule.choices)
        assert any("आ" in prod for prod in dev_rule.choices)

        # Check terminals include Devanagari characters
        assert "अ" in terminals
        assert "आ" in terminals
        assert "a" in terminals
        assert "z" in terminals
