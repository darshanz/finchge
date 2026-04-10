def get_nguyen_grammar(version: int, dim: int) -> str:
    # use protected ops for 7 and 8
    is_protected = version in [7, 8]

    log_op = "plog" if is_protected else "log"
    sqrt_op = "psqrt" if is_protected else "sqrt"

    vars_list = [f"x{i}" for i in range(dim)]
    var_str = " | ".join(vars_list)

    funcs = f"sin | cos | exp | {log_op} | {sqrt_op}"

    # nguyen 9 and 10 are 2d, include power
    if dim > 1:
        funcs += " | pow"

    return f"""
    <expr>  ::= <expr> <op> <expr> | <func>(<expr>) | <var> | <const>
    <op>    ::= + | - | * | /
    <func>  ::= {funcs}
    <var>   ::= {var_str}
    <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 5.0
    """
