import mlflow

from floodit import tracking


def test_start_run_logs_sha_and_data_hash(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with tracking.start_run("unit-test", experiment="test-exp", tracking_uri=uri) as run:
        mlflow.log_param("alpha", 0.1)
        mlflow.log_metric("pr_auc", 0.42)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    data = client.get_run(run_id).data
    assert "git_sha" in data.tags
    assert data.tags["data_hash_train"]  # non-empty
    assert data.params["alpha"] == "0.1"
    assert data.metrics["pr_auc"] == 0.42


def test_git_sha_is_resolved():
    sha = tracking.current_git_sha()
    assert isinstance(sha, str) and len(sha) >= 7
