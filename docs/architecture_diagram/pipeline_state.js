// Hand-edit this file as you progress through the roadmap.
// status: "done" | "in_progress" | "pending"
// architecture_live.html loads this file directly (script tag), so it works
// with a plain double-click open — no local server needed. Edit, save, refresh.
window.PIPELINE_STATE = {
  last_updated: "2026-09-24",
  // Full pipeline complete: generator -> data_loader -> preprocessor -> train (real CV search) -> evaluate -> app.py dashboard.
  // Every step writes to reports/run_manifest.json (git commit, data hash, seed, results) via src/manifest.py.
  nodes: {
    generator:         { status: "done",        note: "src/generate_synthetic_data.py built and verified — 53,305 rows, 0% corrupted per the Step 1 contract." },
    raw_data:          { status: "done",        note: "Original Kaggle CSV found 0% usable (near-total inf corruption) and removed. Replaced with the synthetic same-schema dataset above. Dashboard and README carry an explicit disclaimer about this." },
    data_loader:       { status: "done",        note: "src/data_loader.py resolves paths via .env." },
    preprocessor:      { status: "done",        note: "src/preprocessor.py: feature engineering (harvest/event timestamps), one-hot encoding, StandardScaler. Explicit no-target-leakage test guards X. notebooks/02_preprocessing.ipynb executed end-to-end." },
    processed_split:   { status: "done",        note: "70/15/15 train/val/test split written to data/processed/ as CSV." },
    target_spoilage:   { status: "done",        note: "LinearRegression baseline + LGBMRegressor tuned via RandomizedSearchCV (3-fold CV, 15 iters). Best CV R2 ~0.86." },
    target_efficiency: { status: "done",        note: "LinearRegression baseline + LGBMRegressor tuned via RandomizedSearchCV. Test R2 ~0.80 after raising signal-to-noise in generate_synthetic_data.py." },
    target_quality:    { status: "done",        note: "LinearRegression baseline + LGBMRegressor tuned via RandomizedSearchCV. Test R2 ~0.80." },
    target_croptype:   { status: "done",        note: "Vehicle_Type classification: LogisticRegression baseline + XGBClassifier tuned via RandomizedSearchCV (label-encoded). Test accuracy ~77% after raising signal-to-noise; Bayes-optimal ceiling was computed (~46%) under the original formula before changing it, proving the earlier score was an information limit, not a model deficiency." },
    train:             { status: "done",        note: "src/train.py: real hyperparameter search + CV per target (not hardcoded 'tuned' params). notebooks/03_model_experiments.ipynb executed end-to-end." },
    models:            { status: "done",        note: "8 models (4 targets x baseline/tuned) saved to models/*.joblib." },
    evaluate:          { status: "done",        note: "src/evaluate.py: metrics, residual + residual-vs-predicted plots, feature importance, confusion matrix -> reports/. Streamlit dashboard (app.py) reads these directly." },
    docs:              { status: "done",        note: "Architecture + flow diagrams added. Streamlit dashboard (app.py, with synthetic-data disclaimer + reproducibility panel), pytest suite (18 tests), and GitHub Actions CI (.github/workflows/ci.yml) complete the pipeline." },
    logs:              { status: "done",        note: "Per-step logs (logs/<step>.log) plus reports/run_manifest.json for full run provenance (git commit, data hash, seed, results)." }
  }
};
