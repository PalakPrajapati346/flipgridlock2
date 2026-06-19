"""Explainable AI log lines from model features and prediction context."""

from __future__ import annotations

from datetime import datetime, timezone

from src.train import load_metrics


def build_xai_log(
    impact_score: float,
    impact_band: str,
    telemetry: dict,
    resources: dict,
    feature_row: dict,
    model_name: str,
) -> list[dict]:
    metrics = load_metrics()
    importance = metrics.get("feature_importance", {})
    top_features = list(importance.items())[:5]

    ts = datetime.now(timezone.utc).strftime("%I:%M:%S %p").lstrip("0")
    corridor = feature_row.get("corridor", "unknown")
    cause = feature_row.get("event_cause", "unknown")
    event_type = feature_row.get("event_type", "unplanned")
    priority = feature_row.get("priority", "medium")
    closure = feature_row.get("requires_road_closure", 0)

    lines = [
        {
            "time": ts,
            "type": "forecast",
            "icon": "📈",
            "message": (
                f"{model_name.replace('_', ' ').title()} Forecast: "
                f"{impact_band} congestion on {corridor}. "
                f"ETA to complete halt: {telemetry['halt_eta_minutes']} mins."
            ),
        },
        {
            "time": ts,
            "type": "reasoning",
            "icon": "🧠",
            "message": _reasoning_line(
                impact_score, impact_band, corridor, cause, event_type, priority, closure, top_features
            ),
        },
    ]

    if resources["diversion_level"] != "none":
        pct = {"advisory": 15, "partial": 40, "full": 60}.get(resources["diversion_level"], 25)
        lines.append(
            {
                "time": ts,
                "type": "action",
                "icon": "🚦",
                "message": (
                    f"Diverting ~{pct}% volume to alternate routes on {corridor} "
                    f"prevents queue breaching {telemetry['gridlock_queue_veh']} veh threshold."
                ),
            }
        )

    if resources["manpower"] >= 6:
        lines.append(
            {
                "time": ts,
                "type": "action",
                "icon": "🚑",
                "message": (
                    f"Deploy {resources['manpower']} personnel + {resources['barricades']} barricades. "
                    f"Est. clearance {resources['estimated_clearance_minutes']} min."
                ),
            }
        )

    return lines


def _reasoning_line(
    score: float,
    band: str,
    corridor: str,
    cause: str,
    event_type: str,
    priority: str,
    closure: int,
    top_features: list[tuple[str, float]],
) -> str:
    drivers = []
    if priority in ("high", "low"):
        drivers.append(f"priority={priority}")
    if cause not in ("unknown", ""):
        drivers.append(f"cause={cause}")
    if event_type == "planned":
        drivers.append("planned event window")
    if closure:
        drivers.append("road closure required")

    hist = ""
    if top_features:
        names = ", ".join(f.replace("_", " ") for f, _ in top_features[:3])
        hist = f" Key model drivers: {names}."

    driver_text = ", ".join(drivers) if drivers else "corridor congestion pattern"
    return (
        f"XAI Reasoning: {band} impact ({score}/100) driven by {driver_text} on {corridor}.{hist}"
    )
