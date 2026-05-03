from typer.testing import CliRunner

from finchge.cli import main as cli_main

runner = CliRunner()


def test_check_config_exits_success_when_validation_passes(monkeypatch):
    monkeypatch.setattr(cli_main, "validate_config_file", lambda path: True)

    result = runner.invoke(cli_main.app, ["check-config", "config.yaml"])

    assert result.exit_code == 0


def test_check_config_exits_failure_when_validation_fails(monkeypatch):
    monkeypatch.setattr(cli_main, "validate_config_file", lambda path: False)

    result = runner.invoke(cli_main.app, ["check-config", "bad.yaml"])

    assert result.exit_code == 1


def test_check_grammar_exits_success_when_validation_passes(monkeypatch):
    monkeypatch.setattr(cli_main, "validate_grammar_file", lambda path: True)

    result = runner.invoke(cli_main.app, ["check-grammar", "grammar.bnf"])

    assert result.exit_code == 0


def test_check_grammar_exits_failure_when_validation_fails(monkeypatch):
    monkeypatch.setattr(cli_main, "validate_grammar_file", lambda path: False)

    result = runner.invoke(cli_main.app, ["check-grammar", "bad.bnf"])

    assert result.exit_code == 1


def test_version_option_prints_package_version():
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0
    assert "finchGE" in result.output


def test_doctor_reports_environment_and_templates():
    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0
    assert "finchGE doctor" in result.output
    assert "finchGE version:" in result.output
    assert "Python version:" in result.output
    assert "Platform:" in result.output
    assert "Core dependencies:" in result.output
    assert "Optional dependencies:" in result.output
    assert "Package resources: ok" in result.output
    assert "Templates:" in result.output
    assert "basic: finchGE Example Project - StringMatch" in result.output
