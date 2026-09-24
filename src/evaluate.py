"""Evaluation: metrics, baseline-vs-tuned comparison, residual plots, feature
importance, and a confusion matrix for the classification target. Reads the
models saved by src/train.py and the splits saved by src/preprocessor.py.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from preprocessor import CLASSIFICATION_TARGET, REGRESSION_TARGETS
from train import load_processed

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

VARIANTS = ["baseline", "tuned"]


def load_models(model_dir: Path | None = None) -> dict:
    """Load all target/variant models saved by src/train.py."""
    model_dir = Path(model_dir or (PROJECT_ROOT / os.getenv("MODEL_DIR", "models/")))
    models = {}
    for target in REGRESSION_TARGETS + [CLASSIFICATION_TARGET]:
        models[target] = {}
        for variant in VARIANTS:
            path = model_dir / f"{target}_{variant}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found — run src/train.py first.")
            models[target][variant] = joblib.load(path)
    return models


def evaluate_regression(models: dict, X, y) -> pd.DataFrame:
    """MAE / RMSE / R2 per target/variant."""
    rows = []
    for target in REGRESSION_TARGETS:
        for variant in VARIANTS:
            preds = models[target][variant].predict(X)
            rows.append({
                "target": target,
                "variant": variant,
                "MAE": mean_absolute_error(y[target], preds),
                "RMSE": root_mean_squared_error(y[target], preds),
                "R2": r2_score(y[target], preds),
            })
    return pd.DataFrame(rows)


def evaluate_classification(models: dict, X, y) -> pd.DataFrame:
    """Accuracy / macro-F1 per variant for the classification target."""
    rows = []
    for variant in VARIANTS:
        preds = models[CLASSIFICATION_TARGET][variant].predict(X)
        rows.append({
            "target": CLASSIFICATION_TARGET,
            "variant": variant,
            "accuracy": accuracy_score(y[CLASSIFICATION_TARGET], preds),
            "f1_macro": f1_score(y[CLASSIFICATION_TARGET], preds, average="macro"),
        })
    return pd.DataFrame(rows)


def plot_residuals(models: dict, X, y, out_dir: Path) -> None:
    """Per regression target, tuned model: actual-vs-predicted scatter plus a
    residual-vs-predicted plot (checks for heteroscedasticity / systematic bias)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in REGRESSION_TARGETS:
        preds = models[target]["tuned"].predict(X)
        actual = y[target]
        residuals = actual - preds

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(actual, preds, s=6, alpha=0.3)
        lims = [min(actual.min(), preds.min()), max(actual.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target} — tuned model, actual vs. predicted")
        fig.tight_layout()
        fig.savefig(out_dir / f"residuals_{target}.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(preds, residuals, s=6, alpha=0.3)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{target} — residual vs. predicted")
        fig.tight_layout()
        fig.savefig(out_dir / f"residual_vs_predicted_{target}.png", dpi=120)
        plt.close(fig)

        logger.info("Saved residual plots for %s", target)


def plot_feature_importance(models: dict, feature_names: list, out_dir: Path) -> None:
    """Feature importance bar chart per target, tuned model (skips models without importances)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in REGRESSION_TARGETS + [CLASSIFICATION_TARGET]:
        model = models[target]["tuned"]
        inner = getattr(model, "model", model)  # unwrap LabelEncodedClassifier
        importances = getattr(inner, "feature_importances_", None)
        if importances is None:
            continue

        series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(6, 5))
        series[::-1].plot.barh(ax=ax)
        ax.set_title(f"{target} — tuned model feature importance (top 15)")
        fig.tight_layout()
        fig.savefig(out_dir / f"feature_importance_{target}.png", dpi=120)
        plt.close(fig)
        logger.info("Saved feature importance plot for %s", target)


def plot_confusion_matrix(models: dict, X, y, out_dir: Path) -> None:
    """Confusion matrix heatmap for the classification target, tuned model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = models[CLASSIFICATION_TARGET]["tuned"]
    labels = sorted(y[CLASSIFICATION_TARGET].unique())
    preds = model.predict(X)
    cm = confusion_matrix(y[CLASSIFICATION_TARGET], preds, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{CLASSIFICATION_TARGET} — tuned model confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_Vehicle_Type.png", dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix")


def run_evaluation(reports_dir: Path | None = None) -> dict:
    """Run the full evaluation suite on the held-out test split and write reports."""
    reports_dir = Path(reports_dir or (PROJECT_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = load_processed()
    models = load_models()

    X_test, y_test = splits["X_test"], splits["y_test"]

    reg_metrics = evaluate_regression(models, X_test, y_test)
    clf_metrics = evaluate_classification(models, X_test, y_test)

    reg_metrics.to_csv(reports_dir / "regression_metrics.csv", index=False)
    clf_metrics.to_csv(reports_dir / "classification_metrics.csv", index=False)
    logger.info("Wrote regression_metrics.csv and classification_metrics.csv")

    plot_residuals(models, X_test, y_test, reports_dir / "figures")
    plot_feature_importance(models, X_test.columns.tolist(), reports_dir / "figures")
    plot_confusion_matrix(models, X_test, y_test, reports_dir / "figures")

    summary = {
        "regression": reg_metrics.to_dict(orient="records"),
        "classification": clf_metrics.to_dict(orient="records"),
    }
    with open(reports_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary.json")

    return summary


def main():
    from log_utils import setup_logging
    from manifest import write_manifest

    setup_logging("04_evaluate")
    summary = run_evaluation()
    write_manifest(step="evaluate", extra={"metrics": summary})
    logger.info("Evaluation complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
