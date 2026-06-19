"""Load and validate event congestion datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.config import SAMPLE_CSV

BOOL_MAP = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a"}


def _is_null(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip().lower() in NULL_TOKENS


def _parse_bool(value) -> bool:
    if _is_null(value):
        return False
    if isinstance(value, bool):
        return value
    return BOOL_MAP.get(str(value).strip().lower(), False)


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    cleaned = series.map(lambda v: pd.NA if _is_null(v) else v)
    return pd.to_datetime(cleaned, errors="coerce", utc=True)


def load_events(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else SAMPLE_CSV
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return normalize_events(df)


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["latitude", "longitude", "endlatitude", "endlongitude",
                "resolved_at_latitude", "resolved_at_longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["start_datetime", "end_datetime", "resolved_datetime",
                "closed_datetime", "modified_datetime", "created_date"]:
        if col in df.columns:
            df[col] = _parse_datetime_series(df[col])

    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = df["requires_road_closure"].map(_parse_bool)

    for col in ["event_type", "event_cause", "corridor", "priority", "veh_type",
                "zone", "junction", "status", "direction", "police_station"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
            df.loc[df[col] == "", col] = "unknown"

    return df


def save_uploaded_csv(file_storage, dest: Path) -> pd.DataFrame:
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(dest)
    return load_events(dest)
