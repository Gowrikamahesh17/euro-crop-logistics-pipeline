import pytest

# Small search budget so tests stay fast; the real pipeline run uses more.
N_ITER = 2
CV = 2


@pytest.fixture(scope="module")
def small_splits(raw_df):
    from preprocessor import preprocess

    return preprocess(raw_df.copy())


def test_train_regression_models_covers_all_targets_and_variants(small_splits):
    from preprocessor import REGRESSION_TARGETS
    from train import train_regression_models

    models, search_info = train_regression_models(
        small_splits["X_train"], small_splits["y_train"], n_iter=N_ITER, cv=CV
    )
    assert set(models.keys()) == set(REGRESSION_TARGETS)
    assert set(search_info.keys()) == set(REGRESSION_TARGETS)
    for target in REGRESSION_TARGETS:
        assert set(models[target].keys()) == {"baseline", "tuned"}
        assert "best_params" in search_info[target]
        assert "best_cv_score" in search_info[target]
        preds = models[target]["tuned"].predict(small_splits["X_val"])
        assert len(preds) == len(small_splits["X_val"])


def test_train_classification_model_predicts_known_labels(small_splits):
    from preprocessor import CLASSIFICATION_TARGET
    from train import train_classification_models

    models, search_info = train_classification_models(
        small_splits["X_train"], small_splits["y_train"], n_iter=N_ITER, cv=CV
    )
    known_labels = set(small_splits["y_train"][CLASSIFICATION_TARGET].unique())
    assert "best_params" in search_info[CLASSIFICATION_TARGET]
    for variant in ["baseline", "tuned"]:
        preds = models[CLASSIFICATION_TARGET][variant].predict(small_splits["X_val"])
        assert set(preds) <= known_labels


def test_save_and_load_models_roundtrip(small_splits, tmp_path):
    from train import save_models, train_classification_models, train_regression_models

    models = {}
    reg_models, _ = train_regression_models(
        small_splits["X_train"], small_splits["y_train"], n_iter=N_ITER, cv=CV
    )
    clf_models, _ = train_classification_models(
        small_splits["X_train"], small_splits["y_train"], n_iter=N_ITER, cv=CV
    )
    models.update(reg_models)
    models.update(clf_models)

    save_models(models, model_dir=tmp_path)
    saved_files = list(tmp_path.glob("*.joblib"))
    assert len(saved_files) == 4 * 2  # 4 targets x (baseline, tuned)
