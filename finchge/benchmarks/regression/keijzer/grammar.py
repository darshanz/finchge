def get_keijzer_grammar(dim: int) -> str:
    vars_str = " | ".join([f"x{i}" for i in range(dim)])
    funcs = "sin | cos | exp | log | sqrt | abs | sinh | cosh | asinh"

    if dim == 2:
        funcs += " | pow"

    return f"""
    <expr>  ::= <expr> <op> <expr> | <func>(<expr>) | <var> | <const>
    <op>    ::= + | - | * | /
    <func>  ::= {funcs}
    <var>   ::= {vars_str}
    <const> ::= 0.1 | 0.3 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 | 10.0 | 30.0
    """
