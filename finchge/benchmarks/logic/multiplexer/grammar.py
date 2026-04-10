def get_logic_grammar(n_bits: int) -> str:
    vars_list = " | ".join([f"x{i}" for i in range(n_bits)])  # x[0..N-1]

    return f"""
    <expr> ::= <if> | <and> | <or> | <not> | <var>
    <if>   ::= if(<expr>, <expr>, <expr>)
    <and>  ::= and(<expr>, <expr>)
    <or>   ::= or(<expr>, <expr>)
    <not>  ::= not(<expr>)
    <var>  ::= {vars_list}
    """
