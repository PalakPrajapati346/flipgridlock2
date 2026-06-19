"""Flask web application for event congestion forecasting."""

from __future__ import annotations

import os
import time
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from src.config import ROOT, SAMPLE_CSV
from src.data_loader import load_events
from src.predict import historical_summary, predict_event
from src.resource_model import load_resource_metrics
from src.telemetry import load_corridor_stats
from src.train import load_metrics

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

# Force CSS/JS reload by injecting a timestamp version
@app.context_processor
def inject_version():
    return dict(version=int(time.time()))

def _dataset_path() -> Path:
    override = app.config.get("DATASET_PATH")
    if override:
        return Path(override)
    return SAMPLE_CSV

@app.route("/")
def index():
    df = load_events(_dataset_path())
    summary = historical_summary(df)
    metrics = load_metrics()
    return render_template(
        "index.html",
        summary=summary,
        metrics=metrics,
        dataset=str(_dataset_path()),
    )

@app.route("/dashboard")
def dashboard():
    df = load_events(_dataset_path())
    summary = historical_summary(df)
    return render_template("dashboard.html", summary=summary, dataset=str(_dataset_path()))

@app.route("/forecast")
def forecast_page():
    df = load_events(_dataset_path())
    corridors = sorted(df["corridor"].dropna().unique().tolist())
    causes = sorted(df["event_cause"].dropna().unique().tolist())
    zones = sorted(df["zone"].dropna().unique().tolist())
    corridor_stats = load_corridor_stats(df)
    return render_template(
        "forecast.html",
        corridors=corridors,
        causes=causes,
        zones=zones,
        corridor_stats=corridor_stats,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    result = predict_event(payload)
    return jsonify(result)

@app.route("/api/summary")
def api_summary():
    df = load_events(_dataset_path())
    return jsonify(historical_summary(df))

if __name__ == "__main__":
    # Updated to use the port assigned by the environment (e.g., Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)