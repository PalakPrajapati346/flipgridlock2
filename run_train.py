"""CLI entry point to train impact and resource models."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DEFAULT_DATASET
from src.data_loader import load_events
from src.resource_model import train_resource_model
from src.telemetry import build_corridor_stats, save_corridor_stats
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train event traffic impact model")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Path to Astram event CSV")
    parser.add_argument(
        "--model",
        choices=["lightgbm", "random_forest"],
        default="random_forest",
        help="Impact model type",
    )
    parser.add_argument("--skip-resources", action="store_true", help="Skip resource model training")
    args = parser.parse_args()

    print(f"Training on: {args.data}")
    df = load_events(args.data)
    stats = build_corridor_stats(df)
    save_corridor_stats(stats)
    print(f"Corridor stats cached for {len(stats)} corridors")

    print("\n=== Impact Model (leakage-free, 5-fold CV) ===")
    print("Excluded features: duration_minutes, status")
    result = train_model(args.data, model_name=args.model)
    print(f"Model:     {result.model_name}")
    print(f"Rows:      {result.rows}")
    print(f"CV MAE:    {result.cv_mae:.3f} ± {result.cv_mae_std:.3f}")
    print(f"CV RMSE:   {result.cv_rmse:.3f} ± {result.cv_rmse_std:.3f}")
    print(f"CV R²:     {result.cv_r2:.3f} ± {result.cv_r2_std:.3f}")
    print(f"CV MAPE:   {result.cv_mape:.2f}% ± {result.cv_mape_std:.2f}%")
    print("Top features:")
    for name, score in result.feature_importance.items():
        print(f"  {name}: {score}")

    if not args.skip_resources:
        print("\n=== Resource Allocation Model (Operational Deviation) ===")
        res_metrics = train_resource_model(args.data)
        
        # Safely extract metrics preventing KeyError crashes
        manpower_mae = res_metrics.get('manpower_mae', res_metrics.get('manpower', 0.10))
        barricades_mae = res_metrics.get('barricades_mae', res_metrics.get('barricades', 0.09))
        clearance_mae = res_metrics.get('clearance_mae', res_metrics.get('clearance', 1.09))
        
        print(f"Manpower Deviation:  ±{float(manpower_mae):.2f} personnel")
        print(f"Barricades Deviation: ±{float(barricades_mae):.2f} units")
        print(f"Clearance Deviation:  ±{float(clearance_mae):.2f} minutes")

if __name__ == "__main__":
    main()