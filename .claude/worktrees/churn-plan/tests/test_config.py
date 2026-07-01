from floodit import config


def test_n_jobs_is_capped_not_all_cores():
    # Never -1: leave CPU for other processes (user constraint).
    assert config.N_JOBS != -1
    assert 1 <= config.N_JOBS <= 4


def test_seed_and_columns_present():
    assert config.RANDOM_SEED == 42
    assert config.LABEL_COLUMN == "churned"
    assert len(config.NUMERICAL_COLUMNS) == 11
    assert config.CATEGORICAL_COLUMNS == ["country_name", "device_os", "device_lang"]


def test_njobs_env_override(monkeypatch):
    import importlib

    monkeypatch.setenv("FLOODIT_N_JOBS", "3")
    importlib.reload(config)
    assert config.N_JOBS == 3
    monkeypatch.delenv("FLOODIT_N_JOBS")
    importlib.reload(config)
