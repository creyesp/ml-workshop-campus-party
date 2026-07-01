from sklearn.model_selection import train_test_split

from src.config.settings import RANDOM_STATE, TEST_SIZE
from src.data.source import load_raw_data, load_test_data
from src.train.evaluation import evaluate_model
from src.train.registry import generate_run_id, save_artifact
from src.train.training import train_model


def run_train(
    model_name: str,
    model_params: dict | None = None,
    most_common_thr: float | None = None,
    use_test_split: bool = True,
) -> dict:
    if use_test_split:
        df = load_raw_data()
        x = df.drop(columns=[df.columns[-1]])
        y = df[df.columns[-1]]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        x_train, y_train = load_raw_data().iloc[:, :-1], load_raw_data().iloc[:, -1]
        x_test, y_test = load_test_data()

    pipeline = train_model(
        x_train=x_train,
        y_train=y_train,
        model_name=model_name,
        model_params=model_params,
        most_common_thr=most_common_thr,
    )
    train_metrics = evaluate_model(pipeline, x_train, y_train)
    test_metrics = evaluate_model(pipeline, x_test, y_test)
    run_id = generate_run_id()
    save_artifact(
        pipeline=pipeline,
        metrics={"train": train_metrics, "test": test_metrics},
        run_id=run_id,
        model_name=model_name,
        extra_metadata={
            "model_params": model_params,
        },
    )
    return {
        "run_id": run_id,
        "model_name": model_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
