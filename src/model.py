"""
FPL Model Module

This module contains the FPLModel class responsible for training and predicting
Fantasy Premier League player points using Gradient Boosting Regressor.
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logger = logging.getLogger(__name__)


class FPLModel:
    """
    Fantasy Premier League prediction model using Gradient Boosting Regressor.
    
    This class handles data preparation, model training, validation, and prediction
    following SOLID principles with clear separation of concerns.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the FPL model with configuration.
        
        Args:
            config_path: Optional path to config.yaml file. If None, uses default path.
        """
        self.config = self._load_config(config_path)
        self.model = self._initialize_model()
        self.features = self.config['model']['features']
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional path to config file.
            
        Returns:
            Dictionary containing model configuration.
            
        Raises:
            FileNotFoundError: If config file cannot be found.
            yaml.YAMLError: If config file is invalid YAML.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise
    
    def _initialize_model(self) -> GradientBoostingRegressor:
        """
        Initialize Gradient Boosting Regressor with configured parameters.
        
        Returns:
            Configured GradientBoostingRegressor instance.
        """
        model_config = self.config['model']
        return GradientBoostingRegressor(
            n_estimators=model_config['n_estimators'],
            learning_rate=model_config['learning_rate'],
            max_depth=model_config['max_depth'],
            random_state=model_config['random_state']
        )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training and prediction.
        
        Performs one-hot encoding for positions and ensures all required
        features are present with proper data types.
        
        Args:
            df: Input DataFrame with player data.
            
        Returns:
            DataFrame with encoded features ready for model.
        """
        df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
        
        # Ensure all required features exist
        for feature in self.features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Fill missing values with 0
        df_encoded[self.features] = df_encoded[self.features].fillna(0)
        
        return df_encoded
    
    def prepare_training_data(
        self, 
        df_encoded: pd.DataFrame, 
        original_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by filtering players and calculating target variable.
        
        Args:
            df_encoded: DataFrame with encoded features.
            original_df: Original DataFrame with player information.
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target variable.
        """
        min_minutes = self.config['model']['min_minutes_for_training']
        minutes_per_game = self.config['model']['minutes_per_game']
        
        train_data = df_encoded[df_encoded['minutes'] > min_minutes].copy()
        train_data['pts_per_90'] = (
            (train_data['total_points'] / train_data['minutes']) * minutes_per_game
        )
        
        # Handle infinite and NaN values
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=['pts_per_90'])
        
        X = train_data[self.features]
        y = train_data['pts_per_90']
        
        return X, y
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on provided training data.
        
        Args:
            X_train: Training feature matrix.
            y_train: Training target variable.
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info(f"Model trained successfully on {len(X_train)} samples")
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            raise
    
    def validate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        original_df: pd.DataFrame
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Validate the model on test data and calculate metrics.
        
        Args:
            X_test: Test feature matrix.
            y_test: Test target variable.
            original_df: Original DataFrame for joining player information.
            
        Returns:
            Tuple of (metrics_dict, validation_dataframe).
        """
        try:
            y_pred_test = self.model.predict(X_test)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test),
                'feature_importance': dict(
                    zip(self.features, self.model.feature_importances_)
                )
            }
            
            # Create validation DataFrame with predictions
            validation_df = X_test.copy()
            validation_df['Actual_Points'] = y_test
            validation_df['Predicted_Points'] = y_pred_test
            validation_df['Error'] = (
                validation_df['Actual_Points'] - validation_df['Predicted_Points']
            )
            
            # Join player information
            validation_df = validation_df.join(
                original_df[['web_name', 'team_name', 'position', 'price']],
                how='left'
            )
            
            logger.info(
                f"Validation completed - RMSE: {metrics['rmse']:.3f}, "
                f"MAE: {metrics['mae']:.3f}, R2: {metrics['r2']:.3f}"
            )
            
            return metrics, validation_df
            
        except Exception as e:
            logger.error(f"Error during model validation: {e}", exc_info=True)
            raise
    
    def predict(self, df_encoded: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for all players in the dataset.
        
        Args:
            df_encoded: DataFrame with encoded features.
            
        Returns:
            Series containing predicted points per 90 minutes.
        """
        try:
            predictions = self.model.predict(df_encoded[self.features])
            logger.info(f"Predictions generated for {len(predictions)} players")
            return pd.Series(predictions, index=df_encoded.index)
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise
    
    def train_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        Complete workflow: prepare data, train model, validate, and predict.
        
        This method orchestrates the full pipeline while delegating specific
        tasks to dedicated methods following Single Responsibility Principle.
        
        Args:
            df: Input DataFrame with player data.
            
        Returns:
            Tuple of (predictions_dataframe, metrics_dict, validation_dataframe).
        """
        logger.info("Starting model training and prediction workflow")
        
        try:
            original_df = df.copy()
            
            # Step 1: Prepare features
            df_encoded = self.prepare_features(df)
            
            # Step 2: Prepare training data
            X, y = self.prepare_training_data(df_encoded, original_df)
            
            if len(X) == 0:
                raise ValueError("No training data available after filtering")
            
            # Step 3: Split data for validation
            test_size = self.config['model']['test_size']
            random_state = self.config['model']['random_state']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Step 4: Train model
            self.train(X_train, y_train)
            
            # Step 5: Validate model
            metrics, validation_df = self.validate(X_test, y_test, original_df)
            
            # Step 6: Generate predictions for all players
            df_encoded['predicted_xP_per_90'] = self.predict(df_encoded)
            
            # Restore original position column
            df_encoded['position'] = original_df['position']
            
            logger.info("Model training and prediction workflow completed successfully")
            
            return df_encoded, metrics, validation_df
            
        except Exception as e:
            logger.error(
                f"Error in train_and_predict workflow: {e}",
                exc_info=True
            )
            raise
