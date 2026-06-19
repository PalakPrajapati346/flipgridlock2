"""Resource deployment recommendations — heuristic baseline + learned model hook."""

from __future__ import annotations


def recommend_resources_heuristic(
    score: float, requires_closure: bool, corridor: str, event_type: str
) -> dict:
    is_corridor = corridor not in ("non-corridor", "unknown", "")
    is_planned = event_type == "planned"

    manpower = max(2, int(round(score / 12)))
    if requires_closure:
        manpower += 2
    if is_corridor:
        manpower += 2
    if is_planned:
        manpower += 3

    barricades = 0
    if requires_closure:
        barricades = 6 + int(score // 15)
    elif score >= 50:
        barricades = 4
    elif score >= 25:
        barricades = 2

    diversion_level = "none"
    alternate_routes = []
    if score >= 75 or (requires_closure and is_corridor):
        diversion_level = "full"
        alternate_routes = [
            "Activate parallel arterial",
            "Open service road bypass",
            "Signal retiming on adjacent junctions",
        ]
    elif score >= 40 or requires_closure:
        diversion_level = "partial"
        alternate_routes = ["Lane merge guidance", "Temporary U-turn at next junction"]
    elif score >= 20:
        diversion_level = "advisory"
        alternate_routes = ["Digital message board alert"]

    tow_units = 1 if score >= 30 else 0
    if requires_closure and event_type == "unplanned":
        tow_units = max(tow_units, 1)

    return {
        "manpower": manpower,
        "barricades": barricades,
        "diversion_level": diversion_level,
        "alternate_routes": alternate_routes,
        "tow_units": tow_units,
        "estimated_clearance_minutes": int(max(15, score * 0.9)),
        "source": "heuristic",
    }


def recommend_resources(
    score: float,
    requires_closure: bool,
    corridor: str,
    event_type: str,
    feature_row: dict | None = None,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
) -> dict:
    if feature_row and cat_cols and num_cols:
        from src.resource_model import predict_resources

        learned = predict_resources(feature_row, cat_cols, num_cols)
        if learned:
            if requires_closure and learned["barricades"] < 4:
                learned["barricades"] = 4
            return learned
    return recommend_resources_heuristic(score, requires_closure, corridor, event_type)
