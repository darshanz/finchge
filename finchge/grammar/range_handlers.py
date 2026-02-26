import re
from abc import ABC, abstractmethod
from typing import Optional


class RangeHandler(ABC):
    """
    Base class for all range handlers.
    This is the interface for different types of range handlers.
    There are several range handlers currently supported by finchGE [BNFGrammarParser][finchge.grammar.parser.BNFGrammarParser].
    To add custom range support new range handler can be created and should register by calling.

     `parser.range_registry.register(MyCustomRangeHandler())`

    All range handlers must  inherit from RangeHandler.


    """

    def __init__(self, priority: int = 0):
        self.priority = priority  # Higher priority gets checked first

    @abstractmethod
    def can_handle(self, token: str) -> bool:
        """Check if this handler can process the token."""
        pass

    @abstractmethod
    def expand(self, token: str) -> list[str]:
        """Expand the range into individual symbols."""
        pass

    @property
    @abstractmethod
    def pattern(self) -> str:
        """Return regex pattern that matches this range type."""
        pass


class CharRangeHandler(RangeHandler):
    """Handles character ranges: `'a'..'z'` or `'a'..'z' step 2`"""

    def __init__(self) -> None:
        super().__init__(priority=10)  # High priority for character ranges

    def can_handle(self, token: str) -> bool:
        return bool(re.fullmatch(r"'[^']+'\.\.'[^']+'(?:\s+step\s+\d+)?", token))

    def expand(self, token: str) -> list[str]:
        match = re.match(r"'([^']*)'\.\.'([^']*)'(?:\s+step\s+(\d+))?", token)
        if not match:
            return []

        start_char, end_char, step = match.groups()
        step = int(step) if step else 1

        # Check if these are single characters
        if len(start_char) == 1 and len(end_char) == 1:
            start_code, end_code = ord(start_char), ord(end_char)

            if start_code <= end_code:
                items = [chr(c) for c in range(start_code, end_code + 1, step)]
            else:
                items = [chr(c) for c in range(start_code, end_code - 1, -step)]
        else:
            try:
                start_val, end_val = int(start_char), int(end_char)
                if start_val <= end_val:
                    items = [str(i) for i in range(start_val, end_val + 1, step)]
                else:
                    items = [str(i) for i in range(start_val, end_val - 1, -step)]
            except ValueError:
                return [token]

        expanded = []
        for i, item in enumerate(items):
            expanded.append(str(item))
            if i < len(items) - 1:
                expanded.append("|")
        return expanded

    @property
    def pattern(self) -> str:
        return r"'[^']+'\.\.'[^']+'(?:\s+step\s+\d+)?"


class NumericRangeHandler(RangeHandler):
    """Handles numeric ranges: `10..50`, `-5..5`, `10..50 step 5`"""

    def __init__(self) -> None:
        super().__init__(priority=9)  # High priority, but after char ranges

    def can_handle(self, token: str) -> bool:
        return bool(re.fullmatch(r"-?\d+\.\.-?\d+(?:\s+step\s+-?\d+)?", token))

    def expand(self, token: str) -> list[str]:
        match = re.match(r"(-?\d+)\.\.(-?\d+)(?:\s+step\s+(-?\d+))?", token)
        if not match:
            return []

        start, end, step = match.groups()
        start, end = int(start), int(end)
        step = int(step) if step else 1

        # Smart range generation
        if step > 0:
            if start <= end:
                items = list(range(start, end + 1, step))
            else:
                items = list(range(start, end - 1, -step))
        else:  # step < 0
            if start >= end:
                items = list(range(start, end - 1, step))
            else:
                items = list(range(start, end + 1, -step))

        expanded = []
        for i, item in enumerate(items):
            expanded.append(str(item))
            if i < len(items) - 1:
                expanded.append("|")
        return expanded

    @property
    def pattern(self) -> str:
        return r"-?\d+\.\.-?\d+(?:\s+step\s+-?\d+)?"


class CharClassHandler(RangeHandler):
    """Handles character classes: `[a-z]`, `[A-Za-z0-9_]`"""

    def __init__(self) -> None:
        super().__init__(priority=8)

    def can_handle(self, token: str) -> bool:
        return token.startswith("[") and token.endswith("]")

    def expand(self, token: str) -> list[str]:
        content = token[1:-1]

        items: list[str] = []
        i = 0
        while i < len(content):
            if (
                i + 2 < len(content)
                and content[i + 1] == "-"
                and content[i] != "-"
                and content[i + 2] != "-"
            ):
                start, end = content[i], content[i + 2]
                if ord(start) <= ord(end):
                    items.extend(chr(c) for c in range(ord(start), ord(end) + 1))
                i += 3
            else:
                items.append(content[i])
                i += 1

        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)

        expanded = []
        for idx, item in enumerate(unique_items):
            expanded.append(item)
            if idx < len(unique_items) - 1:
                expanded.append("|")
        return expanded

    @property
    def pattern(self) -> str:
        return r"\[[^\]]+\]"


class SymolicVariableHandler(RangeHandler):
    """Handles individual variables like `x0`, `x1`, etc in symbolic regression, supporting easy evaluation with SymPy like libraries."""

    def __init__(self) -> None:
        super().__init__(priority=50)

    def can_handle(self, token: str) -> bool:
        import re

        # Match patterns like x0, x1, x2, etc.
        return bool(re.fullmatch(r"x\d+", token))

    def expand(self, token: str) -> list[str]:
        # Single variables don't need expansion - they're terminals
        return [token]

    @property
    def pattern(self) -> str:
        return r"x\d+"


class SymbolicVariableRangeHandler(RangeHandler):
    """
    Handles variable ranges for evaluation .

    Replaces ArraySliceRangeHandler.
    Example:
    `x[0..4]` expands to `x0, x1, x2, x3, x4`
    OR
    `x0..4` expands to `x0, x1, x2, x3, x4`
    """

    def __init__(self) -> None:
        super().__init__(priority=52)  # highest

    def can_handle(self, token: str) -> bool:
        import re

        # Match patterns like x[0..4] or x0..4 or x[0..10 step 2]
        patterns = [
            r"x\[\s*\d+\.\.\d+(?:\s+step\s+\d+)?\s*\]",  # x[0..4]
            r"x\d+\.\.\d+(?:\s+step\s+\d+)?",  # x0..4
        ]
        return any(re.fullmatch(pattern, token) for pattern in patterns)

    def expand(self, token: str) -> list[str]:
        import re

        # Extract the range part (supports both formats)
        match = None

        # Try format 1: x[0..4]
        match = re.match(r"x\[\s*(\d+\.\.\d+(?:\s+step\s+\d+)?)\s*\]", token)

        # Try format 2: x0..4
        if not match:
            match = re.match(r"x(\d+\.\.\d+(?:\s+step\s+\d+)?)", token)

        if not match:
            return [token]

        range_expr = match.group(1)

        # For internal ranges we can just use our own NumericRangeHandler
        from .range_handlers import NumericRangeHandler

        num_handler = NumericRangeHandler()

        if not num_handler.can_handle(range_expr):
            return [token]

        # Get expanded numbers
        numbers = [item for item in num_handler.expand(range_expr) if item != "|"]

        # Create SymPy supported variables like x0, x1, x2, so even if the grammar uses numpy array slice style,
        # SymbolicRegression class accepts both and uses sympy style internally
        result = []
        for i, num in enumerate(numbers):
            result.append(f"x{num}")  # x0
            if i < len(numbers) - 1:
                result.append("|")

        return result

    @property
    def pattern(self) -> str:
        return (
            r"(?:x\[\s*\d+\.\.\d+(?:\s+step\s+\d+)?\s*\]|x\d+\.\.\d+(?:\s+step\s+\d+)?)"
        )


class ArraySliceHandler(RangeHandler):
    """
    Handles x[:, 0], x[:, 1], etc.
    This will be useful if we don't have many columns we can just have a grammar like:
    <var> ::= x[:, 0] | x[:, 1]
    However, when the number of columns is higher we may need ArraySliceRangeHandler.
    """

    def __init__(self) -> None:
        super().__init__(priority=52)  # high priority
        # This is given highest priority
        # as it should apply before other range handler
        # except array slice range handler

    def can_handle(self, token: str) -> bool:
        # Match patterns like x[:, 0], x[:, 1], etc.
        return bool(re.fullmatch(r"x\[\s*:\s*,\s*\d+\s*\]", token))

    def expand(self, token: str) -> list[str]:
        # Array slices do not need to be expanded as they are terminals used during evaluation of expressions.
        # this  handler just acts as detector
        return [token]

    @property
    def pattern(self) -> str:
        return r"x\[\s*:\s*,\s*\d+\s*\]"


class ArraySliceRangeHandler(RangeHandler):
    """
    Handles Array slicer with longer range.
    For example:
    x[:, 0..4] will expand to x[:, 0], x[:, 1], ..., x[:, 4]

    If it's not a range and we just need to pick columns randomly we can use individual array slices.

    """

    def __init__(self) -> None:
        super().__init__(priority=53)  # highest

    def can_handle(self, token: str) -> bool:
        # Match patterns like x[:, 0..4] or x[:, 0..10 step 2]
        import re

        return bool(
            re.fullmatch(r"x\[\s*:\s*,\s*\d+\.\.\d+(?:\s+step\s+\d+)?\s*\]", token)
        )

    def expand(self, token: str) -> list[str]:
        import re

        # Extract the range part
        match = re.match(r"x\[\s*:\s*,\s*(\d+\.\.\d+(?:\s+step\s+\d+)?)\s*\]", token)
        if not match:
            return [token]

        range_expr = match.group(1)

        # Use NumericRangeHandler to expand the range
        from .range_handlers import NumericRangeHandler

        num_handler = NumericRangeHandler()

        if not num_handler.can_handle(range_expr):
            return [token]

        # Get expanded numbers
        numbers = [item for item in num_handler.expand(range_expr) if item != "|"]

        # Create array slices for each number
        result = []
        for i, num in enumerate(numbers):
            result.append(f"x[:, {num}]")
            if i < len(numbers) - 1:
                result.append("|")

        return result

    @property
    def pattern(self) -> str:
        return r"x\[\s*:\s*,\s*\d+\.\.\d+(?:\s+step\s+\d+)?\s*\]"


# Registry


class RangeHandlerRegistry:
    """
    Registry for all range handlers with priority support.
    """

    def __init__(self) -> None:
        self._handlers: list[RangeHandler] = []
        self._pattern_cache: Optional[str] = None

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """
        Register default range handlers in priority order.
        """
        self.register(ArraySliceRangeHandler())  # priority=53
        self.register(ArraySliceHandler())  # priority=52
        self.register(SymbolicVariableRangeHandler())  # priority=51
        self.register(SymolicVariableHandler())  # priority=50
        self.register(CharRangeHandler())  # priority=10
        self.register(NumericRangeHandler())  # priority=9
        self.register(CharClassHandler())  # priority=8

    def register(self, handler: RangeHandler) -> None:
        """Register a new handler, maintaining priority order."""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)
        self._pattern_cache = None

    def get_handler(self, token: str) -> Optional[RangeHandler]:
        """Find a handler that can process the token (checking in priority order)."""
        for handler in self._handlers:
            if handler.can_handle(token):
                return handler
        return None

    def get_combined_pattern(self) -> str:
        """Get a combined regex pattern for all handlers in priority order."""
        if self._pattern_cache:
            return self._pattern_cache

        # Get patterns in priority order
        patterns = [handler.pattern for handler in self._handlers]
        self._pattern_cache = "|".join(f"({pattern})" for pattern in patterns)
        return self._pattern_cache

    def expand_token(self, token: str) -> list[str]:
        """Expand a token using the appropriate handler (checking in priority order)."""
        for handler in self._handlers:
            if handler.can_handle(token):
                return handler.expand(token)

        return [token]
