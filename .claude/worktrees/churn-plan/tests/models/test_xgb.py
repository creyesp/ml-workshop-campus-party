from floodit import config
from floodit.data.load import load_dataset
from floodit.models.xgb import build_xgb


def test_build_xgb_default_uses_capped_njobs():
    model = build_xgb({"max_depth": 3})
    assert model.named_steps["clf"].get_params()["n_jobs"] == config.N_JOBS


def test_build_xgb_njobs_override_for_tuning():
    model = build_xgb({"max_depth": 3}, n_jobs=1)
    assert model.named_steps["clf"].get_params()["n_jobs"] == 1


def test_build_xgb_is_seeded_and_fits():
    df = load_dataset("train", verify=False).head(500)
    y = df["churned"].values
    model = build_xgb({"max_depth": 3, "n_estimators": 20})
    assert model.named_steps["clf"].get_params()["random_state"] == config.RANDOM_SEED
    model.fit(df, y)
    proba = model.predict_proba(df)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
