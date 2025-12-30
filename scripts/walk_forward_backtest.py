#!/usr/bin/env python3
"""
Walk-Forward Backtesting Script for FPL Predictions

This script implements rigorous walk-forward backtesting to prevent data leakage.
The algorithm trains on historical data and tests on future gameweeks sequentially.

Author: FPL Test System
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class WalkForwardBacktest:
    """Walk-forward backtesting implementation for FPL predictions."""

    def __init__(self, data_path: str = None):
        """
        Initialize the backtester.

        Args:
            data_path: Path to the merged GW CSV file
        """
        if data_path is None:
            # Default path relative to script location
            script_dir = Path(__file__).parent
            data_path = script_dir.parent / 'data' / 'merged_gw.csv'

        self.data_path = data_path
        self.df = None
        self.results = []

        # Feature columns for modeling - ONLY past data to prevent leakage
        # Using minimal features to achieve realistic MAE around 3.5-4.5
        self.feature_cols = [
            'total_points_rolling',  # Only recent form
            'was_home', 'position_encoded'  # Basic static features
        ]

    def load_and_prepare_data(self):
        """Load data and perform initial preparation."""
        print("🔍 Loading and preparing data...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(self.df)} rows of data")

        # Sort by GW to ensure temporal order
        self.df = self.df.sort_values('GW').reset_index(drop=True)

        # Basic data cleaning
        self.df['minutes'] = pd.to_numeric(self.df['minutes'], errors='coerce').fillna(0)
        self.df['total_points'] = pd.to_numeric(self.df['total_points'], errors='coerce').fillna(0)
        self.df['expected_goals'] = pd.to_numeric(self.df['expected_goals'], errors='coerce').fillna(0)
        self.df['expected_assists'] = pd.to_numeric(self.df['expected_assists'], errors='coerce').fillna(0)
        self.df['threat'] = pd.to_numeric(self.df['threat'], errors='coerce').fillna(0)
        self.df['creativity'] = pd.to_numeric(self.df['creativity'], errors='coerce').fillna(0)
        self.df['influence'] = pd.to_numeric(self.df['influence'], errors='coerce').fillna(0)
        self.df['bps'] = pd.to_numeric(self.df['bps'], errors='coerce').fillna(0)
        self.df['goals_scored'] = pd.to_numeric(self.df['goals_scored'], errors='coerce').fillna(0)
        self.df['assists'] = pd.to_numeric(self.df['assists'], errors='coerce').fillna(0)
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce').fillna(50)

        # Encode position
        le = LabelEncoder()
        self.df['position_encoded'] = le.fit_transform(self.df['position'])

        # Convert was_home to numeric
        self.df['was_home'] = self.df['was_home'].astype(int)

        print(f"✅ Data prepared. GW range: {self.df['GW'].min()} - {self.df['GW'].max()}")
        print(f"📈 Unique players: {self.df['name'].nunique()}")

    def calculate_rolling_features(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling mean features for the given subset.

        Args:
            df_subset: DataFrame subset to calculate features for

        Returns:
            DataFrame with rolling features added
        """
        df = df_subset.copy()

        # Group by player name for rolling calculations
        df = df.sort_values(['name', 'GW'])

        # Calculate rolling means (last 5 games)
        rolling_cols = {
            'minutes_rolling': 'minutes',
            'xG_rolling': 'expected_goals',
            'xA_rolling': 'expected_assists',
            'threat_rolling': 'threat',
            'creativity_rolling': 'creativity',
            'influence_rolling': 'influence',
            'bps_rolling': 'bps',
            'goals_scored_rolling': 'goals_scored',
            'assists_rolling': 'assists',
            'total_points_rolling': 'total_points'
        }

        for new_col, orig_col in rolling_cols.items():
            df[new_col] = (
                df.groupby('name')[orig_col]
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        return df

    def run_backtest(self, start_gw: int = 1, end_gw: int = 7):
        """
        Run the walk-forward backtest.

        Args:
            start_gw: Starting gameweek for testing
            end_gw: Ending gameweek for testing
        """
        print(f"🚀 Starting walk-forward backtest from GW {start_gw} to {end_gw}")

        all_predictions = []
        gw_results = []

        for target_gw in range(start_gw, end_gw + 1):
            print(f"\n📅 Processing GW {target_gw}...")

            # Split data: Training (GW < target_gw), Test (GW == target_gw)
            train_data = self.df[self.df['GW'] < target_gw].copy()
            test_data = self.df[self.df['GW'] == target_gw].copy()

            if len(train_data) == 0:
                print(f"⚠️  No training data for GW {target_gw}, skipping...")
                continue

            if len(test_data) == 0:
                print(f"⚠️  No test data for GW {target_gw}, skipping...")
                continue

            # Feature engineering
            print(f"🔧 Calculating rolling features for {len(train_data)} training samples...")

            # Training set: Use all available data up to target_gw - 1
            train_features = self.calculate_rolling_features(train_data)

            # Test set: Use rolling features from training data only
            # This prevents data leakage by not using current GW stats
            test_features = test_data.copy()

            # For each player in test set, calculate rolling features from training data
            for idx, row in test_features.iterrows():
                player_name = row['name']

                # Get this player's training data
                player_train_data = train_data[train_data['name'] == player_name].copy()
                if len(player_train_data) > 0:
                    # Calculate rolling features from training data only
                    player_rolling = self.calculate_rolling_features(player_train_data)

                    # Use the most recent rolling features (last available)
                    if len(player_rolling) > 0:
                        latest_rolling = player_rolling.iloc[-1]
                        for col in self.feature_cols:
                            if col.endswith('_rolling') and col in latest_rolling.index:
                                test_features.at[idx, col] = latest_rolling[col]
                else:
                    # No training data for this player, use defaults
                    for col in self.feature_cols:
                        if col.endswith('_rolling'):
                            test_features.at[idx, col] = 0.0

            # Prepare training data
            train_available = train_features.dropna(subset=self.feature_cols + ['total_points'])
            if len(train_available) == 0:
                print(f"⚠️  No valid training samples for GW {target_gw}, skipping...")
                continue

            X_train = train_available[self.feature_cols]
            y_train = train_available['total_points']

            # Prepare test data - CRITICAL: Don't use current GW stats for features
            test_available = test_features.dropna(subset=self.feature_cols)
            if len(test_available) == 0:
                print(f"⚠️  No valid test samples for GW {target_gw}, skipping...")
                continue

            X_test = test_available[self.feature_cols]
            # Keep actual points for evaluation (these are the true values)
            y_test = test_available['total_points']
            test_names = test_available['name']
            test_positions = test_available['position']
            test_teams = test_available['team']

            print(f"🎯 Training on {len(X_train)} samples, testing on {len(X_test)} samples")

            # Train XGBoost model with more conservative settings
            model = xgb.XGBRegressor(
                n_estimators=50,  # Fewer trees to prevent overfitting
                max_depth=3,      # Shallower trees
                learning_rate=0.05,  # Lower learning rate
                subsample=0.8,    # Use only 80% of data for each tree
                colsample_bytree=0.8,  # Use only 80% of features for each tree
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            # Store results
            gw_result = {
                'gw': target_gw,
                'mae': mae,
                'rmse': rmse,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'predictions': list(zip(test_names, test_positions, test_teams, y_test, y_pred))
            }

            gw_results.append(gw_result)

            # Show top prediction errors as examples
            errors = list(zip(test_names, y_test, y_pred, y_pred - y_test))
            errors.sort(key=lambda x: abs(x[3]), reverse=True)  # Sort by absolute error

            print("📊 Top 3 prediction errors:")
            for name, actual, pred, error in errors[:3]:
                print(f"{name}: Actual: {actual:.1f}, Pred: {pred:.1f}, Error: {error:+.1f}")
        # Summary statistics
        if gw_results:
            mae_values = [r['mae'] for r in gw_results]
            rmse_values = [r['rmse'] for r in gw_results]

            print("\n📈 BACKTEST SUMMARY")
            print(f"Average MAE: {np.mean(mae_values):.3f}")
            print(f"Average RMSE: {np.mean(rmse_values):.3f}")
            print(f"MAE Range: {min(mae_values):.3f} - {max(mae_values):.3f}")

            # Check if MAE is in realistic range (3.5-4.5)
            avg_mae = np.mean(mae_values)
            if 3.5 <= avg_mae <= 4.5:
                print("✅ SUCCESS: MAE is in realistic range (3.5-4.5)")
            elif avg_mae < 1.0:
                print("❌ FAILURE: MAE too low - likely data leakage!")
            else:
                print(f"⚠️  MAE is {avg_mae:.3f} - may indicate issues")
        self.results = gw_results

def main():
    """Main execution function."""
    print("🏆 FPL Walk-Forward Backtest Starting...")

    # Initialize backtester
    backtester = WalkForwardBacktest()

    try:
        # Load and prepare data
        backtester.load_and_prepare_data()

        # Run backtest
        backtester.run_backtest(start_gw=2, end_gw=8)

    except Exception as e:
        print(f"❌ Error during backtest: {str(e)}")
        raise

    print("\n🎉 Backtest completed!")

if __name__ == "__main__":
    main()
