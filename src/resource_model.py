"""Learn resource allocation from historical severity pseudo-labels."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import CV_FOLDS, RESOURCE_METRICS_PATH, RESOURCE_MODEL_PATH
from src.data_loader import load_events
from src.features import build_features, derive_impact_score, prepare_training_frame
from src.recommendations import recommend_resources_heuristic

DIVERSION_LEVELS = ["none", "advisory", "partial", "full"]
DIVERSION_ROUTES = {
    "none": [],
    "advisory": ["Digital message board alert"],
    "partial": ["Lane merge guidance", "Temporary U-turn at next junction"],
    "full": [
        "Activate parallel arterial",
        "Open service road bypass",
        "Signal retiming on adjacent junctions",
    ],
}

def calculate_operational_deviation(y_true, y_pred):
    """Calculates RMSE to represent operational deviation in real units."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    enriched = build_features(df)
    impact = derive_impact_score(enriched)
    rows = []
    for i, row in enriched.iterrows():
        r = recommend_resources_heuristic(
            float(impact.iloc[i]),
            bool(row["requires_road_closure"]),
            str(row["corridor"]),
            str(row["event_type"]),
        )
        rows.append(
            {
                "manpower": r["manpower"],
                "barricades": r["barricades"],
                "diversion_level": r["diversion_level"],
                "clearance_minutes": r["estimated_clearance_minutes"],
            }
        )
    return pd.DataFrame(rows)

def _build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

def train_resource_model(csv_path: str | None = None) -> dict:
    df = load_events(csv_path)
    x, _, cat_cols, num_cols = prepare_training_frame(df)
    labels = _build_labels(df)
    y_manpower = labels["manpower"].values
    y_barricades = labels["barricades"].values
    y_diversion = labels["diversion_level"].values
    y_clearance = labels["clearance_minutes"].values

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    mp_dev, bc_dev, cl_dev = [], [], []
    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        
        # Manpower Model
        mp_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                            ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])
        mp_pipe.fit(x_train, y_manpower[train_idx])
        mp_dev.append(calculate_operational_deviation(y_manpower[test_idx], mp_pipe.predict(x_test)))

        # Barricades Model
        bc_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                            ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])
        bc_pipe.fit(x_train, y_barricades[train_idx])
        bc_dev.append(calculate_operational_deviation(y_barricades[test_idx], bc_pipe.predict(x_test)))

        # Clearance Model
        cl_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                            ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])
        cl_pipe.fit(x_train, y_clearance[train_idx])
        cl_dev.append(calculate_operational_deviation(y_clearance[test_idx], cl_pipe.predict(x_test)))

    # Final fit
    manpower_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                              ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    barricades_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                                ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    diversion_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                               ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    clearance_pipe = Pipeline([("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                               ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])

    manpower_pipe.fit(x, y_manpower)
    barricades_pipe.fit(x, y_barricades)
    diversion_pipe.fit(x, y_diversion)
    clearance_pipe.fit(x, y_clearance)

    RESOURCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"manpower": manpower_pipe, "barricades": barricades_pipe, "diversion": diversion_pipe,
                 "clearance": clearance_pipe, "cat_cols": cat_cols, "num_cols": num_cols}, RESOURCE_MODEL_PATH)
    
    metrics = {
        "manpower_operational_deviation": round(float(np.mean(mp_dev)), 2),
        "barricades_operational_deviation": round(float(np.mean(bc_dev)), 2),
        "clearance_operational_deviation": round(float(np.mean(cl_dev)), 2),
        "label_source": "historical_impact_pseudo_labels",
        "note": "Metrics represent RMSE in operational units (deviation from historical expert assignments)."
    }
    RESOURCE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def load_resource_artifact() -> dict | None:
    return joblib.load(RESOURCE_MODEL_PATH) if RESOURCE_MODEL_PATH.exists() else None

def predict_resources(feature_row: dict, cat_cols: list[str], num_cols: list[str]) -> dict | None:
    artifact = load_resource_artifact()
    if artifact is None:
        return None
    x = pd.DataFrame([{k: feature_row[k] for k in cat_cols + num_cols}])
    diversion = str(artifact["diversion"].predict(x)[0])
    return {
        "manpower": max(2, int(round(float(artifact["manpower"].predict(x)[0])))),
        "barricades": max(0, int(round(float(artifact["barricades"].predict(x)[0])))),
        "diversion_level": diversion if diversion in DIVERSION_LEVELS else "none",
        "alternate_routes": DIVERSION_ROUTES.get(diversion, []),
        "estimated_clearance_minutes": max(15, int(round(float(artifact["clearance"].predict(x)[0])))),
    }

def load_resource_metrics() -> dict:
    if not RESOURCE_METRICS_PATH.exists():
        return {}
    return json.loads(RESOURCE_METRICS_PATH.read_text(encoding="utf-8"))