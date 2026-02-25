import pytest

from finchge.config.config import FinchConfig, Keys


def test_geconfig_from_ini(tmp_path):
    ini = tmp_path / "ge_config.ini"
    ini.write_text("""
        [ge]
        codon_size = 127

        [experiment]
        verbose = True
        """)

    cfg = FinchConfig.from_ini(str(ini))

    assert cfg.ge[Keys.CODON_SIZE] == 127
    assert cfg.experiment[Keys.VERBOSE] is True


def test_geconfig_from_yaml(tmp_path):
    yaml_file = tmp_path / "ge_config.yaml"
    yaml_file.write_text("""
        ge:
          codon_size: 127
        experiment:
          verbose: true
        """)

    cfg = FinchConfig.from_yaml(str(yaml_file))

    assert cfg.ge[Keys.CODON_SIZE] == 127
    assert cfg.experiment[Keys.VERBOSE] is True


def test_invalid_extension():
    with pytest.raises(ValueError):
        FinchConfig.from_file("config.txt")


def test_copy_with_update():
    cfg = FinchConfig.from_dict({"ge": {Keys.MUTATION_PROBABILITY: 0.01}})

    new_cfg = cfg.copy(update={"ge": {Keys.MUTATION_PROBABILITY: 0.1}})

    assert cfg.ge[Keys.MUTATION_PROBABILITY] == 0.01
    assert new_cfg.ge[Keys.MUTATION_PROBABILITY] == 0.1


def test_copy_is_deep():
    cfg = FinchConfig.from_dict({"ge": {Keys.CODON_SIZE: 127}})
    new_cfg = cfg.copy()

    new_cfg.ge[Keys.CODON_SIZE] = 255
    assert cfg.ge[Keys.CODON_SIZE] == 127


def test_type_parsing_ini(tmp_path):
    ini = tmp_path / "ge.ini"
    ini.write_text("""
        [experiment]
        verbose = True
        num_generations = 100
        mutation_probability = 0.01
        """)

    cfg = FinchConfig.from_ini(str(ini))

    assert isinstance(cfg.experiment[Keys.VERBOSE], bool)
    assert isinstance(cfg.experiment[Keys.NUM_GENERATIONS], int)
    assert isinstance(cfg.experiment[Keys.MUTATION_PROBABILITY], float)


def test_missing_section_is_empty(tmp_path):
    ini = tmp_path / "ge.ini"
    ini.write_text("[other_section]\ncodon_size = 127")

    cfg = FinchConfig.from_ini(str(ini))

    assert cfg.ge == {}
    assert cfg.experiment == {}


def test_from_file_autodetect(tmp_path):
    yaml_file = tmp_path / "ge_params.yaml"
    yaml_file.write_text("ge:\n  codon_size: 127")

    cfg = FinchConfig.from_file(str(yaml_file))
    assert cfg.ge[Keys.CODON_SIZE] == 127


def test_from_file_multiple_configs(tmp_path):
    (tmp_path / "ge_params.yaml").write_text("ge: {}")
    (tmp_path / "ge_params.ini").write_text("[ge]")

    with pytest.raises(RuntimeError):
        FinchConfig.from_file()
