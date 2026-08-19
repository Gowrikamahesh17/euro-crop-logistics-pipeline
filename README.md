# 🌾 Euro Crop Agricultural Logistics — End-to-End ML Pipeline

An end-to-end Machine Learning pipeline that predicts supply chain efficiency, crop spoilage risk, quality maintenance, and transit classifications across European agricultural supply networks.

---

## 📌 Project Architecture & Workflow

![System Architecture](docs/architecture_diagram/system_architecture.svg)

**[▶ Open the interactive version](docs/architecture_diagram/architecture_live.html)** — clone the repo and open `docs/architecture_diagram/architecture_live.html` in a browser for a live, click-to-explore diagram: node colors reflect real build progress (done / in progress / pending, read from `pipeline_state.js`), and hovering or clicking a node shows its role and current status. GitHub's README renderer strips scripts, so this interactive view only runs when opened locally, not embedded on this page.

### Planned Work Flow

![Planned Work Flow](docs/architecture_diagram/planned_work_flow.svg)

---

## 📊 Planned Results Dashboard

Once the pipeline produces real model outputs, a **Streamlit dashboard** will be built to explore them interactively — per-target metrics, residual plots, feature importance, and model comparisons (baseline vs. tuned), reading directly from `models/*.joblib` and `src/evaluate.py` outputs. This is deliberately not built yet: it will land once Steps 4–6 produce metrics worth showing, and lives outside this repo's static `docs/` diagrams since it needs a Python runtime rather than a hosted static page.

---

## 🧪 Data Notes

The original Kaggle dataset ([Euro Crop Agricultural Logistics Dataset](https://www.kaggle.com/datasets/datasetengineer/euro-crop-agricultural-logistics-dataset)) was audited in Step 1 (`notebooks/01_eda_inspection.ipynb`) and found to be structurally unusable — near-universal `float64` overflow corruption across 18 of 23 numeric columns, with 0 fully clean rows and no recoverable signal. Full evidence and the tested-and-disconfirmed recovery attempt are documented in the notebook.

It was replaced with a **synthetic, same-schema dataset** (`src/generate_synthetic_data.py`, 53,305 rows, seeded via `.env`): features drawn from bounded, physically-motivated distributions, with all 4 targets computed as formulas over their relevant features plus noise — giving the modeling steps genuine, learnable relationships rather than corrupted or purely random data. Re-running the same Step 1 audit against it confirms 0% corruption across every column.