"""
Model validation script for FPL points regression.

Usage:
    python scripts/validate_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# XGBoost is required for the validation step
try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "xgboost package is missing. Install with `pip install xgboost`."
    ) from exc


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as file:
        import yaml

        return yaml.safe_load(file) or {}


def load_feature_list(feature_path: Path) -> List[str]:
    with open(feature_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    defensive = payload.get("defensive", [])
    offensive = payload.get("offensive", [])
    return sorted(set(defensive + offensive))


def prepare_dataset(
    data_path: Path, feature_path: Path
) -> Tuple[pd.DataFrame, pd.Series, List[str], str]:
    df = pd.read_csv(data_path)
    target_candidates = ["total_points", "event_points"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Neither 'total_points' nor 'event_points' found in dataset.")

    target_series = pd.to_numeric(df[target_col], errors="coerce")
    df = df.loc[target_series.notna()].copy()
    target_series = target_series.loc[df.index]

    feature_list = load_feature_list(feature_path)
    available_features = [f for f in feature_list if f in df.columns]
    missing = sorted(set(feature_list) - set(available_features))
    if missing:
        print(f"[INFO] Missing {len(missing)} features ignored: {missing}")
    if not available_features:
        raise ValueError("No requested features found in dataset.")

    X = (
        df[available_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = target_series.astype(float)

    return X, y, available_features, target_col


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def format_report(
    label: str, metrics: Dict[str, float], prefix: str = "--- Model Performance Report ---"
) -> str:
    lines = [
        prefix,
        f"Target: {label}",
        f"RMSE: {metrics['rmse']:.3f} (Lower is better)",
        f"MAE:  {metrics['mae']:.3f} (Lower is better)",
        f"R2:   {metrics['r2']:.3f} (Higher is better)",
        "-" * 32,
    ]
    return "\n".join(lines)


def try_risk_adjustment(
    meta_df: pd.DataFrame, base_preds: np.ndarray, y_true: pd.Series
) -> Tuple[Dict[str, float] | None, pd.Series | None]:
    """
    Apply rotation risk penalties from optimizer logic when possible.
    """
    try:
        from src.optimizer import Optimizer

        optimizer = Optimizer()
        meta_df = meta_df.copy()

        minutes_risk = optimizer.calculate_minutes_risk(meta_df)
        meta_df["minutes_risk"] = minutes_risk

        def _risk_multiplier(row: pd.Series) -> float:
            multiplier = 1.0
            if "team_name" in row and row["team_name"] in optimizer.risk_teams:
                multiplier *= optimizer.team_penalty_multiplier
            if row.get("minutes_risk", 0.0) > optimizer.risk_threshold:
                multiplier *= optimizer.risk_coefficient
            return float(multiplier)

        if "risk_multiplier" in meta_df.columns:
            meta_df["_risk_multiplier"] = pd.to_numeric(
                meta_df["risk_multiplier"], errors="coerce"
            ).fillna(1.0)
        else:
            meta_df["_risk_multiplier"] = meta_df.apply(_risk_multiplier, axis=1)

        adjusted_preds = base_preds * meta_df["_risk_multiplier"].to_numpy()
        metrics = compute_metrics(y_true, adjusted_preds)
        return metrics, meta_df["_risk_multiplier"]
    except Exception as exc:
        print(f"[INFO] Risk adjustment skipped: {exc}")
        return None, None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "all_players.csv"
    feature_path = project_root / "data" / "feature_list.json"
    config_path = project_root / "config.yaml"

    config = load_config(config_path)
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    X, y, features, target_col = prepare_dataset(data_path, feature_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_params = {
        "n_estimators": model_cfg.get("n_estimators", 200),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "max_depth": model_cfg.get("max_depth", 4),
        "subsample": model_cfg.get("subsample", 0.9),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.9),
        "random_state": model_cfg.get("random_state", 42),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    base_metrics = compute_metrics(y_test, y_pred)

    # Top feature importances
    importances = list(zip(features, model.feature_importances_))
    top_features = sorted(importances, key=lambda x: x[1], reverse=True)[:5]

    print(format_report(target_col, base_metrics))
    print("Top 5 Features:")
    for idx, (fname, score) in enumerate(top_features, start=1):
        print(f"{idx}. {fname} ({score:.4f})")

    # Optional risk-adjusted evaluation using optimizer logic
    meta_test = X_test.copy()
    if hasattr(X_test, "index"):
        try:
            full_df = pd.read_csv(data_path)
            meta_test = full_df.loc[X_test.index]
        except Exception:
            pass

    risk_metrics, multipliers = try_risk_adjustment(meta_test, y_pred, y_test)
    if risk_metrics:
        delta = risk_metrics["rmse"] - base_metrics["rmse"]
        print("\n--- Risk-Adjusted Report ---")
        print(f"RMSE: {risk_metrics['rmse']:.3f} (Δ {delta:+.3f})")
        print(f"MAE:  {risk_metrics['mae']:.3f}")
        print(f"R2:   {risk_metrics['r2']:.3f}")
        print("--------------------------------")
    else:
        print("\n[INFO] Risk-adjusted evaluation not produced.")


if __name__ == "__main__":
    main()

