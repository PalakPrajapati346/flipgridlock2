from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# UPDATED: Use a relative path so it works on Render and locally without absolute paths
DEFAULT_DATASET = DATA_DIR / "Astram_event_data.csv"
SAMPLE_CSV = DEFAULT_DATASET if DEFAULT_DATASET.exists() else DATA_DIR / "sample_events.csv"

MODEL_PATH = MODEL_DIR / "impact_model.joblib"
RESOURCE_MODEL_PATH = MODEL_DIR / "resource_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RESOURCE_METRICS_PATH = MODEL_DIR / "resource_metrics.json"

# Features available BEFORE event resolution (no leakage)
CATEGORICAL_COLS = [
    "event_type",
    "event_cause",
    "corridor",
    "priority",
    "veh_type",
    "zone",
    "junction",
    "direction",
    "police_station",
]

NUMERIC_COLS = [
    "latitude",
    "longitude",
    "endlatitude",
    "endlongitude",
    "requires_road_closure",
    "start_hour",
    "start_dow",
    "start_month",
    "is_weekend",
    "is_peak_hour",
    "has_vehicle",
    "segment_length_km",
]

# Excluded from training — unknown at forecast time or post-event outcome
LEAKED_COLS = ["duration_minutes", "status"]

CV_FOLDS = 5