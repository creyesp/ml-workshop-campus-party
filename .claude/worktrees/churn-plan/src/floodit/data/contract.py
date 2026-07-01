"""Data contract: assert the loaded frame matches spec assumptions.

Run before any feature/model work. Encodes spec §2 (upstream exclusions hold)
and §6 (stable class balance). Raises AssertionError on the first violation.
"""
import pandas as pd

from floodit import config


def validate_contract(df: pd.DataFrame) -> None:
    # --- Schema: required columns present ---------------------------------
    required = set(
        config.NUMERICAL_COLUMNS
        + config.CATEGORICAL_COLUMNS
        + [config.LABEL_COLUMN, "is_enable", "bounced", "user_pseudo_id"]
    )
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    # --- Count features: non-null, integer-valued, non-negative -----------
    for col in config.NUMERICAL_COLUMNS:
        assert df[col].notna().all(), f"{col} has nulls"
        assert (df[col] >= 0).all(), f"{col} has negative values"
        assert (df[col] == df[col].astype("int64")).all(), f"{col} not integer-valued"

    # --- Label: binary 0/1 ------------------------------------------------
    assert set(df[config.LABEL_COLUMN].unique()) <= {0, 1}, "churned not in {0,1}"

    # --- Spec §2 upstream exclusions still hold ---------------------------
    assert (df["is_enable"] == 1).all(), "is_enable=0 rows present (should be excluded)"
    assert (df["bounced"] == 0).all(), "bounced=1 rows present (should be excluded)"

    # --- Categoricals are string-like (nulls allowed -> imputed downstream)
    for col in config.CATEGORICAL_COLUMNS:
        non_null = df[col].dropna()
        assert non_null.map(lambda v: isinstance(v, str)).all(), f"{col} not string"

    # --- Class balance stable (spec §6: ~23% churn) -----------------------
    prevalence = df[config.LABEL_COLUMN].mean()
    assert 0.20 <= prevalence <= 0.26, f"churn prevalence {prevalence:.3f} out of range"
