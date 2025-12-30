"""
Realistic backtest script for GW18 using vaastav/Fantasy-Premier-League data.

Workflow:
- Download merged gameweek CSV for the current season (fallback across seasons).
- Sort by GW, split train (GW < 18) vs test (GW == 18).
- Align features to `data/feature_list.json`, filling missing columns with 0.
- Train a single XGBoost regressor to predict `total_points`.
- Report RMSE/MAE and top-5 actual scorers in GW18 with predictions.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# Season candidates in case the primary link returns 404.
SEASON_CANDIDATES: List[str] = ["2024-25", "2023-24", "2022-23"]
BASE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "data" / "feature_list.json"


def fetch_merged_gw() -> Tuple[pd.DataFrame, str, str]:
    """Download merged_gw.csv, trying season candidates until one succeeds."""
    errors = []
    for season in SEASON_CANDIDATES:
        url = BASE_URL.format(season=season)
        logger.info("Downloading data for season %s ...", season)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            logger.info("Download ok for %s", season)
            try:
                return pd.read_csv(StringIO(resp.text)), season, url
            except Exception as exc:
                logger.warning(
                    "Standard CSV parse failed for %s (%s); retrying with python engine/skip bad lines",
                    season,
                    exc,
                )
                # More tolerant parsing to handle occasional malformed rows.
                df = pd.read_csv(
                    StringIO(resp.text),
                    engine="python",
                    on_bad_lines="skip",
                )
                logger.info("Parsed with python engine; rows=%d cols=%d", *df.shape)
                return df, season, url
        errors.append((season, resp.status_code))
        logger.warning("Season %s failed with status %s", season, resp.status_code)
    raise RuntimeError(f"Failed to download merged_gw.csv, attempts: {errors}")


def load_feature_union() -> List[str]:
    """Load offensive/defensive feature union from feature_list.json."""
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    defensive = payload.get("defensive", [])
    offensive = payload.get("offensive", [])
    return sorted(set(defensive + offensive))


def normalize_gw_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a GW column exists (vaastav sometimes uses 'round')."""
    df = df.copy()
    if "GW" not in df.columns:
        if "round" in df.columns:
            df = df.rename(columns={"round": "GW"})
        else:
            raise KeyError("Neither 'GW' nor 'round' column found in dataset.")
    return df


def ensure_position(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a position column (GK/DEF/MID/FWD) for pos_* features."""
    df = df.copy()
    if "position" not in df.columns:
        if "element_type" in df.columns:
            mapper = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            df["position"] = df["element_type"].map(mapper).fillna("UNK")
        else:
            df["position"] = "UNK"
    df["position"] = df["position"].astype(str).str.upper()
    return df


def derive_player_name(row: pd.Series) -> str:
    """Pick the best available player name representation."""
    candidates = [
        "web_name",
        "name",
        "player",
        "player_name",
        "second_name",
        "full_name",
    ]
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    first = row.get("first_name")
    second = row.get("second_name")
    if pd.notna(first) or pd.notna(second):
        return f"{first or ''} {second or ''}".strip()
    return f"Player_{row.name}"


def build_feature_matrix(df: pd.DataFrame, feature_union: Iterable[str]) -> pd.DataFrame:
    """Align dataframe to expected feature set, filling missing columns with 0."""
    df = ensure_position(df)
    df = df.copy()

    # Positional one-hot columns expected in feature list.
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)

    feature_df = pd.DataFrame(index=df.index)
    # Minimal mapping: direct match preferred; fall back aliases if provided.
    aliases: Dict[str, str] = {
        "xG_per_90": "expected_goals_per_90",
        "xA_per_90": "expected_assists_per_90",
    }

    for feat in feature_union:
        source = feat
        if source not in df.columns and feat in aliases:
            source = aliases[feat]

        if source in df.columns:
            feature_df[feat] = pd.to_numeric(df[source], errors="coerce")
        else:
            feature_df[feat] = 0.0  # missing feature filled with zero

    feature_df = feature_df.fillna(0.0).astype(float)
    return feature_df


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_union: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Train XGBoost on train_df and predict on test_df."""
    X_train = build_feature_matrix(train_df, feature_union)
    y_train = pd.to_numeric(train_df["total_points"], errors="coerce").fillna(0.0)

    X_test = build_feature_matrix(test_df, feature_union)
    y_test = pd.to_numeric(test_df["total_points"], errors="coerce").fillna(0.0)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse",
    )

    logger.info("Training model on %d samples with %d features", len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    metrics = {"rmse": rmse, "mae": mae}

    results = test_df.copy()
    results["predicted_points"] = preds
    results["actual_points"] = y_test.values
    results["player_name"] = results.apply(derive_player_name, axis=1)
    results["diff"] = results["predicted_points"] - results["actual_points"]

    return results, metrics


def main() -> None:
    df, season, url = fetch_merged_gw()
    feature_union = load_feature_union()

    df = normalize_gw_column(df)
    df = df.sort_values("GW").reset_index(drop=True)

    train_df = df[df["GW"] < 18].copy()
    test_df = df[df["GW"] == 18].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test set is empty. Check GW split or source data integrity."
        )

    results, metrics = train_and_predict(train_df, test_df, feature_union)

    print(f"Data source: {url} (season={season})")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Features used: {len(feature_union)}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")

    top_actual = results.sort_values("actual_points", ascending=False).head(5)
    display_cols = ["player_name", "actual_points", "predicted_points", "diff", "GW"]
    print("\nTop 5 actual performers in GW18 vs model predictions:")
    print(
        top_actual[display_cols]
        .rename(
            columns={
                "player_name": "Player",
                "actual_points": "Actual",
                "predicted_points": "Predicted",
                "diff": "Diff",
            }
        )
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


if __name__ == "__main__":
    main()

