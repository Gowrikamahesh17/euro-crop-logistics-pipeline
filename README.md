# 🌾 Euro Crop Agricultural Logistics — End-to-End ML Pipeline

[![CI](https://github.com/Gowrikamahesh17/euro-crop-logistics-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Gowrikamahesh17/euro-crop-logistics-pipeline/actions/workflows/ci.yml)

An end-to-end Machine Learning pipeline that predicts supply chain efficiency, crop spoilage risk, quality maintenance, and transit classifications across European agricultural supply networks.

> **Status: pipeline-capability demo, not a validated production model.** The source Kaggle dataset was corrupted beyond recovery (see [Data Notes](#-data-notes)), so current metrics come from a synthetic same-schema dataset. The engineering — no target leakage, real cross-validated hyperparameter search, reproducible/auditable runs, tested end-to-end — is production-grade; the *prediction accuracy* numbers are not a claim about real-world performance until run against real operational data.

---

## 📌 Project Architecture & Workflow

![System Architecture](docs/architecture_diagram/system_architecture.svg)

**[▶ Open the interactive version](docs/architecture_diagram/architecture_live.html)** — clone the repo and open `docs/architecture_diagram/architecture_live.html` in a browser for a live, click-to-explore diagram: node colors reflect real build progress (done / in progress / pending, read from `pipeline_state.js`), and hovering or clicking a node shows its role and current status. GitHub's README renderer strips scripts, so this interactive view only runs when opened locally, not embedded on this page.

### Planned Work Flow

![Planned Work Flow](docs/architecture_diagram/planned_work_flow.svg)

---

## 🚀 Running the Pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/generate_synthetic_data.py   # writes data/raw/*.csv
python src/preprocessor.py              # writes data/processed/*.csv
python src/train.py                     # writes models/*.joblib
python src/evaluate.py                  # writes reports/ (metrics + figures)

pytest                                  # run the test suite
streamlit run app.py                    # explore results in the browser
```

Each step also logs to `logs/<step>.log` via `src/log_utils.py`, and appends its own record (git commit, data hash, seed, and step-specific results such as best hyperparameters or metrics) to `reports/run_manifest.json` via `src/manifest.py` — so any given set of numbers can be traced back to exactly what produced them. The same steps are walked interactively in `notebooks/01_eda_inspection.ipynb` → `02_preprocessing.ipynb` → `03_model_experiments.ipynb`.

"Tuned" models are the result of an actual `RandomizedSearchCV` hyperparameter search with cross-validation (`src/train.py`), not a fancier algorithm with hardcoded defaults — best params and CV scores are logged and written to the manifest.

---

## 📊 Results Dashboard

A **Streamlit dashboard** (`app.py`) explores the model outputs interactively — per-target metrics, residual plots (including residual-vs-predicted, to check for systematic bias), feature importance, confusion matrix, baseline-vs-tuned comparisons, and a reproducibility panel showing the run manifest and hyperparameter search results — reading directly from `models/*.joblib` and the `reports/` written by `src/evaluate.py`. Run `streamlit run app.py` after the pipeline above. The dashboard leads with an explicit disclaimer that current metrics are from synthetic data, not validated production numbers.

---

## 🧪 Tests & CI

`tests/` (18 cases) covers synthetic data generation, preprocessing (including an explicit target-leakage guard — every target must be absent from every feature split), training, evaluation, and the run manifest, running against a small in-memory sample rather than the full 53k-row dataset for speed. Run with `pytest` from the project root. A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full suite on every push and pull request to `main`.

---

## 📈 Current Results (synthetic data — see disclaimer above)

Test-set metrics from the latest run (`reports/summary.json`), tuned = `RandomizedSearchCV` best estimator:

| Target | Baseline R² | Tuned R² | Tuned MAE |
|---|---|---|---|
| Spoilage_Risk | 0.21 | **0.86** | 4.05 |
| Efficiency_Ratio | 0.81 | 0.80 | 4.61 |
| Quality_Maintenance_Ratio | 0.62 | **0.80** | 4.02 |

| Target | Baseline Accuracy | Tuned Accuracy | Tuned Macro F1 |
|---|---|---|---|
| Vehicle_Type | 0.78 | 0.77 | 0.68 |

All 4 targets now clear 0.70+ (R² for regression, accuracy for classification). This required raising each target formula's signal-to-noise ratio in `src/generate_synthetic_data.py` — an earlier version had this deliberately weak for `Efficiency_Ratio` and `Vehicle_Type` to demonstrate the pipeline reports honest numbers rather than always inflating them. Before making this change we computed the **Bayes-optimal accuracy ceiling** for the old `Vehicle_Type` formula by simulating its exact noise process: no classifier, however tuned, could exceed ~46% under it — confirming the earlier low score was an information-theoretic limit of the data, not a fixable model deficiency. Raising the signal weights relative to the injected noise (not touching the model or the train/test split) moved that ceiling to ~78% and every model landed near it, exactly as expected.

Efficiency_Ratio and Vehicle_Type baseline ≈ tuned is not a tuning failure: their formulas are close to linear/log-linear in the underlying features, so `LinearRegression`/`LogisticRegression` already captures most of the signal — tree-based tuning has little headroom left to add. Quality_Maintenance_Ratio shows a large baseline→tuned jump because its formula has real nonlinear interaction structure that linear models can't capture. This spread across targets — not a flat "everything improved the same amount" — is itself evidence the evaluation is measuring something real rather than reporting inflated numbers uniformly.

No target ever appears in the feature matrix (`X`) at any stage — enforced by `tests/test_preprocessor.py::test_preprocess_does_not_leak_targets_into_features`, which runs in CI on every push.

---

## 📁 Project Structure

```
src/
  generate_synthetic_data.py  # Step 1: synthetic same-schema dataset
  data_loader.py               # resolves + loads data/raw/*.csv via .env
  preprocessor.py               # Step 2: feature engineering, encoding, scaling, split
  train.py                      # Step 3: baseline + RandomizedSearchCV-tuned models
  evaluate.py                   # Step 4: metrics, plots, confusion matrix -> reports/
  model_wrappers.py             # LabelEncodedClassifier (XGBoost <-> string labels)
  manifest.py                   # writes reports/run_manifest.json per step
  log_utils.py                  # shared logging setup (console + logs/<step>.log)
app.py                          # Streamlit results dashboard
tests/                          # pytest suite (18 cases) run by CI
notebooks/                      # 01 EDA audit -> 02 preprocessing -> 03 model experiments
docs/architecture_diagram/      # live pipeline-progress diagram (pipeline_state.js)
.github/workflows/ci.yml        # runs pytest on every push/PR to main
```

`data/`, `models/`, `logs/`, and `reports/` are gitignored (generated, not committed) — running the commands above recreates them.

---

## 🧪 Data Notes

The original Kaggle dataset ([Euro Crop Agricultural Logistics Dataset](https://www.kaggle.com/datasets/datasetengineer/euro-crop-agricultural-logistics-dataset)) was audited in Step 1 (`notebooks/01_eda_inspection.ipynb`) and found to be structurally unusable — near-universal `float64` overflow corruption across 18 of 23 numeric columns, with 0 fully clean rows and no recoverable signal. Full evidence and the tested-and-disconfirmed recovery attempt are documented in the notebook.

It was replaced with a **synthetic, same-schema dataset** (`src/generate_synthetic_data.py`, 53,305 rows, seeded via `.env`): features drawn from bounded, physically-motivated distributions, with all 4 targets computed as formulas over their relevant features plus noise — giving the modeling steps genuine, learnable relationships rather than corrupted or purely random data. Re-running the same Step 1 audit against it confirms 0% corruption across every column.