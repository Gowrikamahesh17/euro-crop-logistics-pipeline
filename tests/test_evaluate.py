import pytest


@pytest.fixture(scope="module")
def small_setup(raw_df):
    from preprocessor import preprocess
    from train import train_classification_models, train_regression_models

    splits = preprocess(raw_df.copy())
    models = {}
    reg_models, _ = train_regression_models(splits["X_train"], splits["y_train"], n_iter=2, cv=2)
    clf_models, _ = train_classification_models(splits["X_train"], splits["y_train"], n_iter=2, cv=2)
    models.update(reg_models)
    models.update(clf_models)
    return splits, models


def test_evaluate_regression_returns_metrics_for_all_targets(small_setup):
    from evaluate import evaluate_regression
    from preprocessor import REGRESSION_TARGETS

    splits, models = small_setup
    df = evaluate_regression(models, splits["X_test"], splits["y_test"])
    assert set(df["target"]) == set(REGRESSION_TARGETS)
    assert set(df["variant"]) == {"baseline", "tuned"}
    assert {"MAE", "RMSE", "R2"} <= set(df.columns)
    assert (df["MAE"] >= 0).all()


def test_evaluate_classification_returns_metrics(small_setup):
    from evaluate import evaluate_classification

    splits, models = small_setup
    df = evaluate_classification(models, splits["X_test"], splits["y_test"])
    assert len(df) == 2
    assert df["accuracy"].between(0, 1).all()
    assert df["f1_macro"].between(0, 1).all()


def test_run_evaluation_writes_reports(tmp_path, monkeypatch, small_setup):
    from evaluate import run_evaluation
    import train as train_module

    splits, models = small_setup

    monkeypatch.setattr(train_module, "load_processed", lambda: splits)
    monkeypatch.setattr("evaluate.load_processed", lambda: splits)
    monkeypatch.setattr("evaluate.load_models", lambda: models)

    summary = run_evaluation(reports_dir=tmp_path)

    assert (tmp_path / "regression_metrics.csv").exists()
    assert (tmp_path / "classification_metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "figures" / "confusion_matrix_Vehicle_Type.png").exists()
    assert "regression" in summary and "classification" in summary
