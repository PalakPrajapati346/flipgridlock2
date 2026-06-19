"""Train impact model with K-Fold CV and leakage-free features."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import CV_FOLDS, METRICS_PATH, MODEL_PATH
from src.data_loader import load_events
from src.features import prepare_training_frame

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    LGBMRegressor = None


@dataclass
class TrainResult:
    model_name: str
    rows: int
    cv_mae: float
    cv_mae_std: float
    cv_rmse: float
    cv_rmse_std: float
    cv_r2: float
    cv_r2_std: float
    cv_mape: float
    cv_mape_std: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    leakage_notes: list[str] = field(default_factory=list)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.asarray(y_true, dtype=float), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def _build_model(model_name: str):
    if model_name == "lightgbm" and HAS_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def _feature_importance(pipeline: Pipeline, cat_cols: list[str], num_cols: list[str]) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    names = cat_names + num_cols

    if not hasattr(model, "feature_importances_"):
        return {}

    pairs = sorted(zip(names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    return {k: round(float(v), 4) for k, v in pairs[:15]}


def _cross_validate(x: pd.DataFrame, y: pd.Series, cat_cols: list[str], num_cols: list[str], model_name: str) -> dict:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    maes, rmses, r2s, mapes = [], [], [], []

    for train_idx, test_idx in kf.split(x):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(cat_cols, num_cols)),
                ("model", _build_model(model_name)),
            ]
        )
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)

        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(float(np.sqrt(mean_squared_error(y_test, preds))))
        r2s.append(r2_score(y_test, preds))
        mapes.append(_mape(y_test.values, preds))

    return {
        "cv_mae": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_rmse": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_r2": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s)),
        "cv_mape": float(np.mean(mapes)),
        "cv_mape_std": float(np.std(mapes)),
        "cv_folds": CV_FOLDS,
    }


def train_model(csv_path: str | None = None, model_name: str = "random_forest") -> TrainResult:
    df = load_events(csv_path)
    x, y, cat_cols, num_cols = prepare_training_frame(df)

    actual_model = model_name if model_name == "lightgbm" and HAS_LIGHTGBM else "random_forest"
    cv = _cross_validate(x, y, cat_cols, num_cols, actual_model)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(cat_cols, num_cols)),
            ("model", _build_model(actual_model)),
        ]
    )
    pipeline.fit(x, y)
    importance = _feature_importance(pipeline, cat_cols, num_cols)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "cat_cols": cat_cols, "num_cols": num_cols, "model_name": actual_model},
        MODEL_PATH,
    )

    metrics = {
        "model_name": actual_model,
        "rows": len(df),
        "evaluation": "5-fold_cross_validation",
        "leakage_removed": ["duration_minutes", "status"],
        **{k: round(v, 3) for k, v in cv.items() if k != "cv_folds"},
        "cv_folds": CV_FOLDS,
        "feature_importance": importance,
        # Legacy keys for templates — map to CV metrics
        "mae": round(cv["cv_mae"], 3),
        "r2": round(cv["cv_r2"], 3),
        "rmse": round(cv["cv_rmse"], 3),
        "mape": round(cv["cv_mape"], 3),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainResult(
        model_name=actual_model,
        rows=len(df),
        feature_importance=importance,
        leakage_notes=["duration_minutes", "status"],
        **{k: cv[k] for k in ("cv_mae", "cv_mae_std", "cv_rmse", "cv_rmse_std", "cv_r2", "cv_r2_std", "cv_mape", "cv_mape_std")},
    )


def load_trained_artifact() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
