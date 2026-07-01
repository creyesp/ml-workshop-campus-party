"""Tests del pipeline de features."""

from __future__ import annotations

import numpy as np
from src.config.settings import get_settings
from src.features.build import build_feature_pipeline


def test_feature_pipeline_fits_only_on_train(sample_dataframe):
    """El pipeline se ajusta con train y transforma test con la misma forma."""
    data = get_settings().data
    features = sample_dataframe[data.feature_columns]

    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(features.iloc[:150])
    test_transformed = pipeline.transform(features.iloc[150:])

    assert transformed.shape[1] == test_transformed.shape[1]
    assert transformed.shape[0] == 150


def test_unknown_category_is_ignored(sample_dataframe):
    """Una categoría no vista en train no rompe la transformación de test."""
    data = get_settings().data
    features = sample_dataframe[data.feature_columns].copy()

    pipeline = build_feature_pipeline()
    pipeline.fit(features)

    unseen = features.iloc[[0]].copy()
    unseen["country_name"] = "Atlantis"
    result = pipeline.transform(unseen)
    assert not np.isnan(
        result.toarray() if hasattr(result, "toarray") else result
    ).any()
