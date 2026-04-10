def get_koza_grammar() -> str:
    return """
    <expr>  ::= <expr> <op> <expr> | <func>(<expr>) | <var> | <const>
    <op>    ::= + | - | * | /
    <func>  ::= sin | cos | exp | log | sqrt
    <var>   ::= x0
    <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 5.0 | 10.0
    """
