"""Streamlit results dashboard: per-target metrics, residual plots, feature
importance, and baseline-vs-tuned comparison, reading from models/ and reports/
(the outputs of src/train.py and src/evaluate.py).

Run with: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Euro Crop Logistics — Model Results",
    page_icon="🌾",
    layout="wide",
)


@st.cache_data
def load_reports():
    reg = pd.read_csv(REPORTS_DIR / "regression_metrics.csv")
    clf = pd.read_csv(REPORTS_DIR / "classification_metrics.csv")
    with open(REPORTS_DIR / "summary.json") as f:
        summary = json.load(f)
    manifest_path = REPORTS_DIR / "run_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else None
    return reg, clf, summary, manifest


st.title("🌾 Euro Crop Agricultural Logistics — Model Results")
st.caption(
    "Predicting spoilage risk, delivery efficiency, quality maintenance, and "
    "vehicle-type classification across European agricultural supply networks."
)


if not (REPORTS_DIR / "summary.json").exists():
    st.error(
        "No evaluation reports found. Run `python src/evaluate.py` first "
        "(after `src/preprocessor.py` and `src/train.py`)."
    )
    st.stop()

reg_metrics, clf_metrics, summary, run_manifest = load_reports()

def build_report_markdown(reg, clf, summary, manifest):
    lines = [
        "# Euro Crop Logistics — Model Results Report",
        "",
        "## Regression targets",
        "",
        reg.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Classification target — Vehicle Type",
        "",
        clf.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Raw summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    if manifest:
        lines += ["", "## Run manifest", "", "```json", json.dumps(manifest, indent=2), "```"]
    return "\n".join(lines) + "\n"


try:
    report_md = build_report_markdown(reg_metrics, clf_metrics, summary, run_manifest)
except ImportError:  # DataFrame.to_markdown needs `tabulate`
    report_md = None

dl_col1, dl_col2, _ = st.columns([1, 1, 4])
if report_md:
    dl_col1.download_button(
        "⬇️ Report (Markdown)",
        data=report_md,
        file_name="euro_crop_model_report.md",
        mime="text/markdown",
    )
dl_col2.download_button(
    "⬇️ Metrics (CSV)",
    data=pd.concat(
        [reg_metrics.assign(task="regression"), clf_metrics.assign(task="classification")]
    ).to_csv(index=False),
    file_name="euro_crop_metrics.csv",
    mime="text/csv",
)

st.divider()
st.header("📊 Regression targets — Spoilage Risk, Efficiency, Quality")

reg_targets = reg_metrics["target"].unique().tolist()
tabs = st.tabs(reg_targets)

for tab, target in zip(tabs, reg_targets):
    with tab:
        target_df = reg_metrics[reg_metrics["target"] == target].set_index("variant")
        baseline_r2 = target_df.loc["baseline", "R2"]
        tuned_r2 = target_df.loc["tuned", "R2"]

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "R² (tuned)", f"{tuned_r2:.3f}", delta=f"{tuned_r2 - baseline_r2:+.3f} vs baseline"
        )
        col2.metric("MAE (tuned)", f"{target_df.loc['tuned', 'MAE']:.2f}")
        col3.metric("RMSE (tuned)", f"{target_df.loc['tuned', 'RMSE']:.2f}")

        st.dataframe(
            target_df[["MAE", "RMSE", "R2"]].style.format("{:.3f}"),
            use_container_width=True,
        )

        img_col1, img_col2, img_col3 = st.columns(3)
        residual_path = FIGURES_DIR / f"residuals_{target}.png"
        resid_vs_pred_path = FIGURES_DIR / f"residual_vs_predicted_{target}.png"
        importance_path = FIGURES_DIR / f"feature_importance_{target}.png"
        if residual_path.exists():
            img_col1.image(str(residual_path), caption="Actual vs. predicted (tuned model)")
        if resid_vs_pred_path.exists():
            img_col2.image(
                str(resid_vs_pred_path),
                caption="Residual vs. predicted (checks for systematic bias)",
            )
        if importance_path.exists():
            img_col3.image(str(importance_path), caption="Feature importance (tuned model)")

st.divider()
st.header("🚚 Classification target — Vehicle Type")

clf_indexed = clf_metrics.set_index("variant")
col1, col2 = st.columns(2)
col1.metric(
    "Accuracy (tuned)",
    f"{clf_indexed.loc['tuned', 'accuracy']:.3f}",
    delta=f"{clf_indexed.loc['tuned', 'accuracy'] - clf_indexed.loc['baseline', 'accuracy']:+.3f} vs baseline",
)
col2.metric(
    "Macro F1 (tuned)",
    f"{clf_indexed.loc['tuned', 'f1_macro']:.3f}",
    delta=f"{clf_indexed.loc['tuned', 'f1_macro'] - clf_indexed.loc['baseline', 'f1_macro']:+.3f} vs baseline",
)

st.dataframe(clf_indexed[["accuracy", "f1_macro"]].style.format("{:.3f}"), use_container_width=True)

img_col1, img_col2 = st.columns(2)
cm_path = FIGURES_DIR / "confusion_matrix_Vehicle_Type.png"
fi_path = FIGURES_DIR / "feature_importance_Vehicle_Type.png"
if cm_path.exists():
    img_col1.image(str(cm_path), caption="Confusion matrix (tuned model)")
if fi_path.exists():
    img_col2.image(str(fi_path), caption="Feature importance (tuned model)")

st.divider()
st.header("⚖️ Baseline vs. tuned — all targets")

comparison = pd.concat(
    [
        reg_metrics.pivot(index="target", columns="variant", values="R2").rename(
            columns={"baseline": "baseline (R2)", "tuned": "tuned (R2)"}
        ),
    ]
)
st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

with st.expander("Raw metrics (JSON)"):
    st.json(summary)

st.divider()
st.header("🔍 Reproducibility — run manifest")

if run_manifest:

    search_info = run_manifest.get("steps", {}).get("train", {}).get("hyperparameter_search")
    if search_info:
        st.subheader("Hyperparameter search results (RandomizedSearchCV)")
        rows = []
        for target, info in search_info.items():
            rows.append({
                "target": target,
                "scoring": info.get("scoring"),
                "best_cv_score": info.get("best_cv_score"),
                "cv_folds": info.get("cv_folds"),
                "n_iter": info.get("n_iter"),
                "best_params": json.dumps(info.get("best_params")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("Full run manifest (JSON)"):
        st.json(run_manifest)
else:
    st.info("No run_manifest.json found — run the pipeline via src/preprocessor.py, src/train.py, and src/evaluate.py to generate one.")

st.divider()
st.caption(
    "Data: synthetic same-schema replacement for the EuroCrop Kaggle dataset "
    "(see README for why). Pipeline: generate → preprocess → train → evaluate. "
    "CI runs the test suite on every push (see .github/workflows/ci.yml)."
)
