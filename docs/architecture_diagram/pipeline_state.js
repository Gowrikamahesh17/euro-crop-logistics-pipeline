// Hand-edit this file as you progress through the roadmap.
// status: "done" | "in_progress" | "pending"
// architecture_live.html loads this file directly (script tag), so it works
// with a plain double-click open — no local server needed. Edit, save, refresh.
window.PIPELINE_STATE = {
  last_updated: "2026-08-20",
  nodes: {
    generator:         { status: "done",        note: "src/generate_synthetic_data.py built and verified — 53,305 rows, 0% corrupted per the Step 1 contract." },
    raw_data:          { status: "done",        note: "Original Kaggle CSV found 0% usable (near-total inf corruption) and removed. Replaced with the synthetic same-schema dataset above." },
    data_loader:       { status: "done",        note: "src/data_loader.py resolves paths via .env." },
    preprocessor:      { status: "pending",     note: "Not started — next up now that Step 1 has a clean dataset." },
    processed_split:   { status: "pending",     note: "Depends on preprocessor.py." },
    target_spoilage:   { status: "pending",     note: "" },
    target_efficiency: { status: "pending",     note: "" },
    target_quality:    { status: "pending",     note: "" },
    target_croptype:   { status: "pending",     note: "" },
    train:             { status: "pending",     note: "" },
    models:            { status: "pending",     note: "" },
    evaluate:          { status: "pending",     note: "" },
    docs:              { status: "in_progress", note: "Architecture + flow diagrams added." },
    logs:              { status: "done",        note: "Logging engine active since project setup." }
  }
};
