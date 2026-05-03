from finchge.cli.validate import validate_grammar_file


def test_validate_grammar_reports_parsed_summary(tmp_path, capsys):
    grammar_file = tmp_path / "grammar.bnf"
    grammar_file.write_text("<expr> ::= x | y\n", encoding="utf-8")

    assert validate_grammar_file(str(grammar_file)) is True

    output = capsys.readouterr().out
    assert "is valid" in output
    assert "Parsed grammar with 1 rules" in output
    assert "start rule <expr>" in output
    assert "can terminate: True" in output


def test_validate_grammar_rejects_non_terminating_start_rule(tmp_path, capsys):
    grammar_file = tmp_path / "grammar.bnf"
    grammar_file.write_text("<expr> ::= <expr>\n", encoding="utf-8")

    assert validate_grammar_file(str(grammar_file)) is False

    output = capsys.readouterr().out
    assert "is invalid" in output
    assert "Parsed grammar with 1 rules" in output
    assert "start rule <expr>" in output
    assert "can terminate: False" in output


def test_validate_grammar_rejects_parse_errors(tmp_path, capsys):
    grammar_file = tmp_path / "grammar.bnf"
    grammar_file.write_text("<expr> ::= <missing>\n", encoding="utf-8")

    assert validate_grammar_file(str(grammar_file)) is False

    output = capsys.readouterr().out
    assert "Invalid grammar file" in output
