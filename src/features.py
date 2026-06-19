"""Feature engineering and impact score derivation."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from src.config import CATEGORICAL_COLS, LEAKED_COLS, NUMERIC_COLS

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(v == 0 for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def compute_duration_minutes(row: pd.Series) -> float:
    start = row.get("start_datetime")
    end = (
        row.get("resolved_datetime")
        or row.get("closed_datetime")
        or row.get("end_datetime")
        or row.get("modified_datetime")
    )
    if pd.isna(start) or pd.isna(end):
        return np.nan
    minutes = (end - start).total_seconds() / 60.0
    if minutes <= 0:
        return np.nan
    return min(minutes, 24 * 60)

def derive_impact_score(df: pd.DataFrame) -> pd.Series:
    """Target label only — duration used here, NOT as a model input."""
    duration = df.apply(compute_duration_minutes, axis=1).fillna(30.0)
    priority_weight = df["priority"].map({"high": 1.4, "medium": 1.0, "low": 0.7, "unknown": 1.0}).fillna(1.0)
    closure_weight = df["requires_road_closure"].astype(float) * 0.35 + 1.0
    corridor_weight = df["corridor"].apply(
        lambda c: 1.25 if c not in ("non-corridor", "unknown", "") else 1.0
    )
    planned_weight = df["event_type"].map({"planned": 1.1, "unplanned": 1.0}).fillna(1.0)
    raw = duration * priority_weight * closure_weight * corridor_weight * planned_weight
    score = np.clip(raw / 15.0, 1.0, 100.0)
    return score.round(2)

def get_impact_band(score: float) -> str:
    """Uses quartile-based threshold logic for balanced reporting."""
    if score < 25: return "Low"
    if score < 50: return "Moderate"
    if score < 75: return "High"
    return "Critical"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = df["start_datetime"]
    df["start_hour"] = start.dt.hour.fillna(12).astype(int)
    df["start_dow"] = start.dt.dayofweek.fillna(0).astype(int)
    df["start_month"] = start.dt.month.fillna(6).astype(int)
    df["is_weekend"] = (df["start_dow"] >= 5).astype(int)
    df["is_peak_hour"] = df["start_hour"].isin([8, 9, 17, 18, 19]).astype(int)
    df["has_vehicle"] = df.get("veh_type", pd.Series("", index=df.index)).replace("unknown", "").ne("").astype(int)
    df["segment_length_km"] = df.apply(
        lambda r: haversine_km(r["latitude"], r["longitude"], r["endlatitude"], r["endlongitude"]),
        axis=1,
    )
    df["duration_minutes"] = df.apply(compute_duration_minutes, axis=1)
    median_duration = df["duration_minutes"].median()
    if pd.isna(median_duration):
        median_duration = 45.0
    df["duration_minutes"] = df["duration_minutes"].fillna(median_duration)
    df["requires_road_closure"] = df["requires_road_closure"].astype(int)
    return df

def _train_feature_cols(enriched: pd.DataFrame) -> list[str]:
    allowed = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in enriched.columns]
    return [c for c in allowed if c not in LEAKED_COLS]

def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    enriched = build_features(df)
    y = derive_impact_score(enriched)
    feature_cols = _train_feature_cols(enriched)
    x = enriched[feature_cols].copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in x.columns]
    num_cols = [c for c in NUMERIC_COLS if c in x.columns]
    return x, y, cat_cols, num_cols

def row_to_feature_dict(payload: dict) -> dict:
    """Pre-event features only — no duration (unknown at forecast time)."""
    start = pd.to_datetime(payload.get("start_datetime"), errors="coerce", utc=True)
    if pd.isna(start):
        start = pd.Timestamp.utcnow()
    lat = float(payload.get("latitude", 0) or 0)
    lon = float(payload.get("longitude", 0) or 0)
    end_lat = float(payload.get("endlatitude", 0) or 0)
    end_lon = float(payload.get("endlongitude", 0) or 0)
    requires_closure = str(payload.get("requires_road_closure", "false")).lower() in ("true", "1", "yes")

    return {
        "event_type": str(payload.get("event_type", "unplanned")).lower(),
        "event_cause": str(payload.get("event_cause", "unknown")).lower(),
        "corridor": str(payload.get("corridor", "unknown")).lower(),
        "priority": str(payload.get("priority", "medium")).lower(),
        "veh_type": str(payload.get("veh_type", "unknown")).lower() or "unknown",
        "zone": str(payload.get("zone", "unknown")).lower(),
        "junction": str(payload.get("junction", "unknown")).lower(),
        "direction": str(payload.get("direction", "unknown")).lower(),
        "police_station": str(payload.get("police_station", "unknown")).lower(),
        "latitude": lat,
        "longitude": lon,
        "endlatitude": end_lat,
        "endlongitude": end_lon,
        "requires_road_closure": int(requires_closure),
        "start_hour": int(start.hour),
        "start_dow": int(start.dayofweek),
        "start_month": int(start.month),
        "is_weekend": int(start.dayofweek >= 5),
        "is_peak_hour": int(start.hour in (8, 9, 17, 18, 19)),
        "has_vehicle": int(bool(payload.get("veh_type"))),
        "segment_length_km": haversine_km(lat, lon, end_lat, end_lon),
    }

def expected_duration_minutes(payload: dict) -> float:
    """Used for telemetry only — operator estimate, not fed to impact model."""
    return float(payload.get("expected_duration_minutes", 45))