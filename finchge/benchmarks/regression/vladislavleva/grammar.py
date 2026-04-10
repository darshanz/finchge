def get_vlad_grammar(dim: int) -> str:
    vars_list = " | ".join([f"x{i}" for i in range(dim)])

    # Literature standard: 2D uses different constants than high-D
    consts = (
        "0.1 | 0.5 | 1.0 | 1.2 | 2.5" if dim == 2 else "0.1 | 0.5 | 1.0 | 2.0 | 5.0"
    )

    return f"""
    <expr>  ::= <expr> <op> <expr> | <func>(<expr>) | <var> | <const>
    <op>    ::= + | - | * | /
    <func>  ::= sin | cos | exp | plog | psqrt
    <var>   ::= {vars_list}
    <const> ::= {consts}
    """
