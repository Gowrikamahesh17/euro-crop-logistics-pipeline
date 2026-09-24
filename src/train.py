"""Model training: baseline + tuned models for the 3 regression targets and
1 classification target, trained on the preprocessed splits in data/processed/.

"Tuned" means an actual randomized hyperparameter search with cross-validation
(RandomizedSearchCV), not just a fancier algorithm with hardcoded defaults.
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from model_wrappers import LabelEncodedClassifier
from preprocessor import CLASSIFICATION_TARGET, REGRESSION_TARGETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# How many random hyperparameter combinations to try per tuned model, and
# how many CV folds to score each one on. Raised from the original 15/3 for a
# more thorough search now that speed is less of a constraint than accuracy.
N_SEARCH_ITER = int(os.getenv("N_SEARCH_ITER", 40))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

REGRESSION_PARAM_DIST = {
    "n_estimators": randint(100, 700),
    "learning_rate": uniform(0.01, 0.24),
    "num_leaves": randint(15, 100),
    "max_depth": randint(3, 14),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lambda": uniform(0.0, 1.0),
}

CLASSIFICATION_PARAM_DIST = {
    "n_estimators": randint(100, 700),
    "learning_rate": uniform(0.01, 0.24),
    "max_depth": randint(3, 12),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0.0, 0.5),
}


def load_processed(processed_dir: Path | None = None) -> dict:
    """Load X_train/X_val/X_test/y_train/y_val/y_test from data/processed/."""
    processed_dir = Path(
        processed_dir or (PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed/"))
    )
    splits = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run src/preprocessor.py first."
            )
        splits[name] = pd.read_csv(path)
    return splits


def _search(estimator, param_dist, X, y, scoring, n_iter, cv, seed):
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "scoring": scoring,
        "cv_folds": cv,
        "n_iter": n_iter,
    }


def train_regression_models(
    X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = N_SEARCH_ITER, cv: int = CV_FOLDS
) -> tuple[dict, dict]:
    """Train baseline (LinearRegression) + tuned (RandomizedSearchCV over LGBMRegressor)
    models for each of the 3 regression targets. Returns (models, search_info)."""
    models, search_info = {}, {}
    for target in REGRESSION_TARGETS:
        models[target] = {}

        logger.info("Training %s / baseline", target)
        baseline = LinearRegression()
        baseline.fit(X_train, y_train[target])
        models[target]["baseline"] = baseline

        logger.info("Training %s / tuned (RandomizedSearchCV, %d iters, %d-fold CV)", target, n_iter, cv)
        tuned, info = _search(
            LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
            REGRESSION_PARAM_DIST,
            X_train,
            y_train[target],
            scoring="r2",
            n_iter=n_iter,
            cv=cv,
            seed=RANDOM_SEED,
        )
        models[target]["tuned"] = tuned
        search_info[target] = info
        logger.info("%s / tuned best CV R2: %.4f, params: %s", target, info["best_cv_score"], info["best_params"])

    return models, search_info


def train_classification_models(
    X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = N_SEARCH_ITER, cv: int = CV_FOLDS
) -> tuple[dict, dict]:
    """Train baseline (LogisticRegression) + tuned (RandomizedSearchCV over XGBClassifier,
    label-encoded) model for Vehicle_Type. Returns (models, search_info)."""
    y = y_train[CLASSIFICATION_TARGET]
    models = {CLASSIFICATION_TARGET: {}}

    logger.info("Training %s / baseline", CLASSIFICATION_TARGET)
    baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    baseline.fit(X_train, y)
    models[CLASSIFICATION_TARGET]["baseline"] = baseline

    logger.info(
        "Training %s / tuned (RandomizedSearchCV, %d iters, %d-fold CV)",
        CLASSIFICATION_TARGET, n_iter, cv,
    )
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    best_raw, info = _search(
        XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss"),
        CLASSIFICATION_PARAM_DIST,
        X_train,
        y_enc,
        scoring="f1_macro",
        n_iter=n_iter,
        cv=cv,
        seed=RANDOM_SEED,
    )
    tuned = LabelEncodedClassifier(best_raw)
    tuned.encoder = encoder  # reuse the fitted encoder so classes_ match training
    models[CLASSIFICATION_TARGET]["tuned"] = tuned
    logger.info(
        "%s / tuned best CV f1_macro: %.4f, params: %s",
        CLASSIFICATION_TARGET, info["best_cv_score"], info["best_params"],
    )

    return models, {CLASSIFICATION_TARGET: info}


def save_models(models: dict, model_dir: Path | None = None) -> None:
    """Persist each target/variant model to models/<target>_<variant>.joblib."""
    model_dir = Path(model_dir or (PROJECT_ROOT / os.getenv("MODEL_DIR", "models/")))
    model_dir.mkdir(parents=True, exist_ok=True)

    for target, variants in models.items():
        for variant, model in variants.items():
            path = model_dir / f"{target}_{variant}.joblib"
            joblib.dump(model, path)
            logger.info("Saved %s", path)


def main():
    from log_utils import setup_logging
    from manifest import write_manifest

    setup_logging("03_train")
    splits = load_processed()

    models = {}
    search_info = {}

    reg_models, reg_info = train_regression_models(splits["X_train"], splits["y_train"])
    models.update(reg_models)
    search_info.update(reg_info)

    clf_models, clf_info = train_classification_models(splits["X_train"], splits["y_train"])
    models.update(clf_models)
    search_info.update(clf_info)

    save_models(models)
    write_manifest(step="train", extra={"hyperparameter_search": search_info})


if __name__ == "__main__":
    main()
