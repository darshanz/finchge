def get_vlad_grammar(dim: int) -> str:
    vars_str = " | ".join([f"x{i}" for i in range(dim)])

    return f"""
    <e> ::= (<e> + <e>)
          | (<e> - <e>)
          | (<e> * <e>)
          | pdiv(<e>, <e>)
          | pow(<e>, <e>)
          | sin(<e>)
          | cos(<e>)
          | exp(<e>)
          | plog(<e>)
          | psqrt(<e>)
          | abs(<e>)
          | sinh(<e>)
          | cosh(<e>)
          | asinh(<e>)
          | <v>
          | <c>

    <v> ::= {vars_str}

    <c> ::= 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 1.2 | 2.5 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 | 10.0 | 30.0 | pi
    """
