def get_santafe_grammar() -> str:
    return """
           <code> ::= <line> | <code> <line>
           <line> ::= <condition> | <op>
           <condition> ::= ifelse food-ahead (<line>) (<line>)
           <op> ::= turn-left | turn-right | move
           """
