// Hand-edit this file as you progress through the roadmap.
// status: "done" | "in_progress" | "pending"
// architecture_live.html loads this file directly (script tag), so it works
// with a plain double-click open — no local server needed. Edit, save, refresh.
window.PIPELINE_STATE = {
  last_updated: "2026-08-18",
  nodes: {
    raw_data:          { status: "done",        note: "CSV present in data/raw/, 53,305 rows confirmed." },
    data_loader:       { status: "done",        note: "src/data_loader.py resolves paths via .env." },
    preprocessor:      { status: "pending",     note: "Not started — blocked on Step 1 audit conclusions." },
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
