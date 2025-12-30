from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Ensure local src package is importable when running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer import Optimizer
from src.data_loader import DataLoader

DATA_PATH = PROJECT_ROOT / "data" / "all_players.csv"


def _prepare_dataframe(df: pd.DataFrame, optimizer: Optimizer) -> pd.DataFrame:
    """Normalize numeric columns and enforce minimum minutes filter."""
    df = df.copy()

    numeric_cols: List[str] = ["price", "gw19_xP", "risk_multiplier", "minutes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Default to no penalty when the column is missing
    if "risk_multiplier" not in df.columns:
        df["risk_multiplier"] = 1.0

    df["risk_multiplier"] = df["risk_multiplier"].fillna(1.0).clip(lower=0.0)

    min_minutes = optimizer.config["optimizer"].get("min_minutes_for_optimization", 0)
    if "minutes" in df.columns:
        df = df[df["minutes"] >= float(min_minutes)]

    df = df.dropna(subset=["gw19_xP", "price", "position", "team_name", "web_name"])
    return df


def format_table(squad_df: pd.DataFrame) -> str:
    """Return a string table with requested columns."""
    display_df = squad_df[
        [
            "position",
            "web_name",
            "team_name",
            "price",
            "gw19_xP",
            "set_piece_threat",
            "Risk_Penalty",
        ]
    ].rename(
        columns={
            "position": "Position",
            "web_name": "Player Name",
            "team_name": "Team",
            "price": "Price",
            "gw19_xP": "gw19_xP",
            "set_piece_threat": "set_piece_threat",
        }
    )

    return display_df.to_string(
        index=False,
        formatters={
            "Price": lambda x: f"{float(x):.1f}",
            "gw19_xP": lambda x: f"{float(x):.2f}",
            "set_piece_threat": lambda x: f"{float(x):.4f}",
            "Risk_Penalty": lambda x: f"{float(x):.2f}",
        },
    )


def main() -> None:
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        sys.exit(f"Data file not found at {DATA_PATH}")

    # Ensure set_piece_threat exists for display (and as a sanity check)
    if "set_piece_threat" not in df.columns:
        try:
            df = DataLoader().enrich_data(df)
        except Exception:
            # Non-fatal: proceed without the column if enrichment fails
            df["set_piece_threat"] = 0.0

    optimizer = Optimizer()
    df = _prepare_dataframe(df, optimizer)

    squad = optimizer.solve_dream_team(
        df, target_metric="gw19_xP", budget=100.0
    )

    if squad.empty:
        sys.exit("No squad could be generated. Check the input data.")

    squad = squad.sort_values("gw19_xP", ascending=False).reset_index(drop=True)
    squad["Risk_Penalty"] = (1.0 - squad["risk_multiplier"]).clip(lower=0.0)

    total_points = squad["gw19_xP"].sum()
    total_cost = squad["price"].sum()

    captain = squad.iloc[0]
    vice_captain = squad.iloc[1] if len(squad) > 1 else None

    print("=" * 72)
    print("GW19 Optimized Squad (Budget=100.0, Metric=gw19_xP)")
    print("=" * 72)
    print(format_table(squad))
    print("-" * 72)
    print(f"Total Predicted Points: {total_points:.2f}")
    print(f"Total Cost: {total_cost:.1f}")
    print("-" * 72)
    print(
        f"Captain (C): {captain['web_name']} "
        f"({captain['gw19_xP']:.2f} gw19_xP)"
    )
    if vice_captain is not None:
        print(
            f"Vice-Captain (VC): {vice_captain['web_name']} "
            f"({vice_captain['gw19_xP']:.2f} gw19_xP)"
        )


if __name__ == "__main__":
    main()

