"""Data-driven telemetry: speed, queue, CO2, fuel — calibrated from Astram history."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATASET, MODEL_DIR, ROOT
from src.features import build_features, derive_impact_score

STATS_CACHE = MODEL_DIR / "corridor_stats.json"

# Bengaluru arterial free-flow when corridor has no speed samples (km/h)
DEFAULT_FREE_FLOW_KMH = 42.0
IDLE_CO2_KG_PER_VEH_MIN = 0.0067  # ~400 g/hr per idling car
CO2_KG_PER_LITRE_DIESEL = 2.68
DIESEL_PRICE_INR = 92.0


def _cap_speed(s: float) -> float:
    return float(np.clip(s, 5.0, 65.0))

def calculate_economic_savings(idle_co2_kg):
    # Example: ₹92 per liter fuel, 2.68 kg CO2 per liter
    fuel_saved_liters = idle_co2_kg / 2.68
    savings_inr = fuel_saved_liters * 92
    return round(savings_inr, 2)


def _compute_event_speed(row: pd.Series) -> float | None:
    seg = row.get("segment_length_km", 0) or 0
    dur = row.get("duration_minutes", 0) or 0
    if seg > 0.05 and dur > 1:
        return _cap_speed(seg / (dur / 60.0))
    return None


def build_corridor_stats(df: pd.DataFrame) -> dict:
    enriched = build_features(df)
    enriched["impact_score"] = derive_impact_score(enriched)
    enriched["event_speed"] = enriched.apply(_compute_event_speed, axis=1)

    span_days = max(
        (enriched["start_datetime"].max() - enriched["start_datetime"].min()).days, 1
    )

    stats: dict[str, dict] = {}
    for corridor, grp in enriched.groupby("corridor"):
        speeds = grp["event_speed"].dropna()
        valid = speeds[(speeds >= 15) & (speeds <= 55)]
        if len(valid) >= 3:
            free_flow = float(np.percentile(valid, 60))
        elif str(corridor).lower() in ("non-corridor", "unknown", "null", ""):
            free_flow = 32.0
        else:
            free_flow = 40.0

        event_count = len(grp)
        events_per_day = event_count / span_days
        hourly_flow = 900 + min(events_per_day * 45, 1300)

        stats[str(corridor)] = {
            "event_count": int(event_count),
            "avg_impact": round(float(grp["impact_score"].mean()), 2),
            "avg_duration_min": round(float(grp["duration_minutes"].mean()), 1),
            "closure_rate": round(float(grp["requires_road_closure"].mean()), 3),
            "median_speed_kmh": round(float(valid.median()), 1) if len(valid) else None,
            "free_flow_kmh": round(free_flow, 1),
            "hourly_flow_est": round(hourly_flow),
            "lat_min": round(float(grp["latitude"].min()), 6),
            "lat_max": round(float(grp["latitude"].max()), 6),
            "lon_min": round(float(grp["longitude"].min()), 6),
            "lon_max": round(float(grp["longitude"].max()), 6),
        }

    return stats


def save_corridor_stats(stats: dict) -> None:
    STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    STATS_CACHE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_corridor_stats(df: pd.DataFrame | None = None) -> dict:
    if STATS_CACHE.exists():
        return json.loads(STATS_CACHE.read_text(encoding="utf-8"))
    if df is None:
        from src.data_loader import load_events

        df = load_events(DEFAULT_DATASET)
    stats = build_corridor_stats(df)
    save_corridor_stats(stats)
    return stats


def _corridor_key(corridor: str, stats: dict) -> str:
    c = corridor.lower().strip()
    if c in stats:
        return c
    for k in stats:
        if k.lower() == c:
            return k
    return "non-corridor" if "non-corridor" in stats else next(iter(stats))


def compute_telemetry(
    impact_score: float,
    duration_minutes: float,
    corridor: str,
    requires_closure: bool,
    stats: dict | None = None,
) -> dict:
    """Derive live telemetry from ML impact + corridor historical calibration."""
    stats = stats or load_corridor_stats()
    key = _corridor_key(corridor, stats)
    cstat = stats.get(key, stats.get("non-corridor", {}))

    free_flow = cstat.get("free_flow_kmh", DEFAULT_FREE_FLOW_KMH)
    avg_impact = max(cstat.get("avg_impact", 10), 1.0)
    hourly_flow = cstat.get("hourly_flow_est", 1200)

    impact_ratio = impact_score / avg_impact
    closure_penalty = 0.18 if requires_closure else 0.0
    speed_factor = max(0.35, 1.0 - (impact_score / 100) * 0.55 - closure_penalty)
    avg_speed_kmh = round(free_flow * speed_factor, 1)

    delay_minutes = min(120.0, duration_minutes * min(2.0, 0.4 + impact_ratio * 0.45))
    delay_hours = delay_minutes / 60.0
    congestion_factor = 0.12 + 0.88 * (impact_score / 100)
    gridlock_queue = int(round(hourly_flow * delay_hours * congestion_factor))

    idle_minutes = min(delay_minutes, 75.0) * min(1.0, gridlock_queue / max(hourly_flow * 0.12, 1))
    idle_co2_kg = round(min(gridlock_queue * idle_minutes * IDLE_CO2_KG_PER_VEH_MIN, 650.0), 1)
    fuel_litres = idle_co2_kg / CO2_KG_PER_LITRE_DIESEL
    fuel_cost_inr = int(min(round(fuel_litres * DIESEL_PRICE_INR), 18500))

    halt_eta_min = int(max(5, delay_minutes * (impact_score / max(avg_impact, 1))))

    return {
        "avg_speed_kmh": avg_speed_kmh,
        "gridlock_queue_veh": gridlock_queue,
        "idle_co2_kg": idle_co2_kg,
        "fuel_cost_inr": fuel_cost_inr,
        "delay_minutes": round(delay_minutes, 1),
        "halt_eta_minutes": halt_eta_min,
        "corridor_stats_used": {
            "corridor": key,
            "free_flow_kmh": free_flow,
            "hourly_flow_est": hourly_flow,
            "historical_avg_impact": avg_impact,
            "historical_event_count": cstat.get("event_count", 0),
        },
    }


def compute_historical_telemetry(df: pd.DataFrame) -> dict:
    """Fleet-wide telemetry averaged from every historical event."""
    enriched = build_features(df)
    enriched["impact_score"] = derive_impact_score(enriched)
    stats = load_corridor_stats(df)

    impacts = enriched["impact_score"].astype(float).values
    durations = enriched["duration_minutes"].astype(float).values
    corridors = enriched["corridor"].astype(str).values
    closures = enriched["requires_road_closure"].astype(bool).values

    speeds, queues, co2s, fuels, delays = [], [], [], [], []
    for i in range(len(enriched)):
        t = compute_telemetry(impacts[i], durations[i], corridors[i], closures[i], stats)
        speeds.append(t["avg_speed_kmh"])
        queues.append(t["gridlock_queue_veh"])
        co2s.append(t["idle_co2_kg"])
        fuels.append(t["fuel_cost_inr"])
        delays.append(t["delay_minutes"])

    return {
        "avg_speed_kmh": round(float(np.median(speeds)), 1),
        "gridlock_queue_veh": int(round(float(np.median(queues)))),
        "idle_co2_kg": round(float(np.median(co2s)), 1),
        "fuel_cost_inr": int(round(float(np.median(fuels)))),
        "total_idle_co2_kg": round(float(np.sum(co2s)), 1),
        "total_fuel_cost_inr": int(np.sum(fuels)),
        "avg_delay_minutes": round(float(np.median(delays)), 1),
    }


def map_position(latitude: float, longitude: float, corridor: str, stats: dict | None = None) -> dict:
    """Normalize event coords to 0–100% for corridor schematic (not a simulation)."""
    stats = stats or load_corridor_stats()
    key = _corridor_key(corridor, stats)
    cstat = stats.get(key, {})

    lat_min, lat_max = cstat.get("lat_min", latitude - 0.01), cstat.get("lat_max", latitude + 0.01)
    lon_min, lon_max = cstat.get("lon_min", longitude - 0.01), cstat.get("lon_max", longitude + 0.01)

    lat_span = max(lat_max - lat_min, 0.001)
    lon_span = max(lon_max - lon_min, 0.001)

    return {
        "x_pct": round((longitude - lon_min) / lon_span * 100, 1),
        "y_pct": round((1 - (latitude - lat_min) / lat_span) * 100, 1),
        "corridor": key,
    }
