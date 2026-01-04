#!/usr/bin/env python3
"""
Test script for optimizer team constraints
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from optimizer import Optimizer

def test_team_constraints():
    print("🧪 Testing Optimizer Team Constraints...")

    # Initialize components
    data_loader = DataLoader()
    optimizer = Optimizer()

    # Load data
    print("📊 Loading FPL data...")
    df_understat, df_fpl, df_fixtures = data_loader.fetch_all_data()

    if df_fpl.empty:
        print("❌ No FPL data available")
        return

    print(f"✅ Loaded {len(df_fpl)} players")

    # Test optimization
    print("🎯 Running optimization with team constraints...")
    result_df = optimizer.solve_dream_team(df_fpl, target_metric='predicted_xP', budget=100.0)

    if result_df.empty:
        print("❌ Optimization failed")
        return

    # Analyze team distribution
    print("\n📈 Analyzing team distribution in optimized squad:")
    team_counts = result_df['team_name'].value_counts().sort_values(ascending=False)

    max_from_team = team_counts.max()
    violations = team_counts[team_counts > 3]

    print(f"Total players selected: {len(result_df)}")
    print(f"Teams used: {len(team_counts)}")
    print(f"Max players from any team: {max_from_team}")
    print(f"Team constraint violations: {len(violations)}")

    print("\nTeam Distribution:")
    for team, count in team_counts.items():
        status = "❌ VIOLATION!" if count > 3 else "✅ OK"
        print(f"  {team}: {count} players {status}")

    if len(violations) > 0:
        print("\n🚨 CONSTRAINT VIOLATIONS DETECTED!")
        print("Teams with too many players:")
        for team, count in violations.items():
            print(f"  {team}: {count} players (should be ≤ 3)")
        return False
    else:
        print("\n✅ ALL CONSTRAINTS SATISFIED!")
        return True

if __name__ == "__main__":
    success = test_team_constraints()
    sys.exit(0 if success else 1)
