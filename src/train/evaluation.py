import pandas as pd
from sklearn.pipeline import Pipeline

from src.train.metrics import classification_report_dict


def evaluate_model(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict:
    y_pred = pipeline.predict(x)
    y_proba = pipeline.predict_proba(x)[:, 1]
    metrics = classification_report_dict(
        y_true=y.values,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    return metrics
