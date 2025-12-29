"""
FPL Model Module

This module contains the FPLModel class responsible for training and predicting
Fantasy Premier League player points using segmented XGBoost regressors.
"""

import logging
import ast
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# XGBoost - segment bazlı modeller için kullanıyoruz
try:
    from xgboost import XGBRegressor
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "xgboost paketi bulunamadı. Lütfen `pip install xgboost` komutunu çalıştırın."
    ) from e

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
        # Temel ve segment bazlı feature listeleri
        base_features = self.config['model']['features']

        # Savunma tarafı: clean sheet, kurtarış ve xGA odaklı
        self.defensive_features = sorted(set(base_features + [
            'clean_sheets', 'saves', 'goals_conceded', 'goals_conceded_per_90',
            'expected_goals_conceded_per_90', 'minutes', 'volatility_index'
        ]))

        # Ofansif taraf: xG/xA ve tehdit skorları odaklı
        self.offensive_features = sorted(set(base_features + [
            'goals_scored', 'assists', 'threat', 'creativity',
            'expected_goals_per_90', 'expected_assists_per_90',
            'expected_goal_involvements_per_90', 'minutes', 'volatility_index'
        ]))

        # Tüm feature'ları bir araya getirerek encode sürecinde eksik sütunları garanti ediyoruz
        self.features = sorted(set(self.defensive_features + self.offensive_features))

        # XGBoost hiperparametreleri (segment bazlı)
        rs = self.config['model']['random_state']
        self.def_model_params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "random_state": rs,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }
        self.off_model_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "random_state": rs,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }

        # Modeller
        self.def_model = None
        self.off_model = None
    
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
    
    def _ensure_feature_columns(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """Feature listindeki tüm sütunların DataFrame'de var olduğundan emin olur."""
        for feature in feature_list:
            if feature not in df.columns:
                df[feature] = 0
        df[feature_list] = df[feature_list].fillna(0)
        return df

    def _prepare_training_data_segment(
        self,
        df_encoded: pd.DataFrame,
        feature_list: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Segment bazlı eğitim datasını hazırlar."""
        min_minutes = self.config['model']['min_minutes_for_training']
        minutes_per_game = self.config['model']['minutes_per_game']

        train_data = df_encoded[df_encoded['minutes'] > min_minutes].copy()
        train_data['pts_per_90'] = (
            (train_data['total_points'] / train_data['minutes']) * minutes_per_game
        )

        # Handle infinite and NaN values
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=['pts_per_90'])

        train_data = self._ensure_feature_columns(train_data, feature_list)
        X = train_data[feature_list]
        y = train_data['pts_per_90']

        return X, y

    def _train_segment_model(
        self,
        df_segment: pd.DataFrame,
        feature_list: List[str],
        model_params: Dict,
        meta_df: pd.DataFrame
    ) -> Tuple[XGBRegressor, Dict, pd.DataFrame]:
        """Verilen segment (DEF/GK veya MID/FWD) için modeli eğitir ve doğrular."""
        X, y = self._prepare_training_data_segment(df_segment, feature_list)

        if len(X) == 0:
            raise ValueError("Segment için eğitim datası bulunamadı.")

        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(feature_list, model.feature_importances_))
        }

        validation_df = X_test.copy()
        validation_df['Actual_Points'] = y_test
        validation_df['Predicted_Points'] = y_pred_test
        validation_df['Error'] = validation_df['Actual_Points'] - validation_df['Predicted_Points']

        # Oyuncu bilgilerini ekle
        info_cols = ['web_name', 'team_name', 'position', 'price']
        available = [c for c in info_cols if c in meta_df.columns]
        if available:
            validation_df = validation_df.join(meta_df.loc[validation_df.index, available], how='left')

        return model, metrics, validation_df

    def _compute_volatility_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        'volatility_index' özelliğini üretir.
        Son 5 haftadaki puanların standart sapmasını hesaplar; veri yoksa 0 döner.
        """
        candidate_cols = [
            'recent_points', 'points_history', 'history_points',
            'past_points', 'gw_points', 'points_last_five'
        ]

        def _safe_to_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return None
            return None

        def _last5_points(row):
            for col in candidate_cols:
                if col in row.index and pd.notna(row[col]):
                    lst = _safe_to_list(row[col])
                    if lst:
                        numeric = [float(x) for x in lst if pd.notna(x)]
                        if numeric:
                            return numeric[-5:]
            return []

        volatility = []
        for _, r in df.iterrows():
            pts = _last5_points(r)
            if len(pts) >= 2:
                volatility.append(float(np.std(pts)))
            else:
                volatility.append(0.0)

        df['volatility_index'] = volatility
        return df

    @staticmethod
    def _calculate_upside_potential(mean_points: float, volatility: float, alpha: float = 0.35) -> float:
        """
        Upside potential: ortalama + alpha * std. Standart sapma yüksekse tavan puanını yükseltir.
        """
        mean_points = mean_points if pd.notna(mean_points) else 0.0
        volatility = volatility if pd.notna(volatility) else 0.0
        return mean_points + alpha * volatility
    
    def train_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        İki segmentli (Defansif: GK+DEF, Ofansif: MID+FWD) XGBoost modelleri ile
        eğitim/prediction akışı. Ayrıca volatility_index ve upside_potential özelliklerini ekler.
        """
        logger.info("Starting segmented model training and prediction workflow")

        try:
            original_df = df.copy()

            # 0. Volatility feature
            df = self._compute_volatility_index(df.copy())

            # 1. Feature encoding
            df_encoded = self.prepare_features(df)

            # Orijinal bilgileri koru
            df_encoded['position_orig'] = original_df['position']
            df_encoded['web_name'] = original_df['web_name']
            df_encoded['team_name'] = original_df['team_name']
            df_encoded['price'] = original_df['price']

            # 2. Segmentasyonu hazırla
            defensive_mask = df_encoded['position_orig'].isin(['GKP', 'GK', 'DEF'])
            offensive_mask = df_encoded['position_orig'].isin(['MID', 'FWD'])

            df_def = df_encoded[defensive_mask].copy()
            df_off = df_encoded[offensive_mask].copy()

            # 3. Defensive model
            self.def_model, def_metrics, def_val = self._train_segment_model(
                df_def, self.defensive_features, self.def_model_params, df_encoded
            )

            # 4. Offensive model
            self.off_model, off_metrics, off_val = self._train_segment_model(
                df_off, self.offensive_features, self.off_model_params, df_encoded
            )

            # 5. Tahminler
            df_encoded['predicted_xP_per_90'] = 0.0
            df_encoded['predicted_xP_def'] = np.nan
            df_encoded['predicted_xP_off'] = np.nan
            df_encoded['upside_potential'] = np.nan

            # Defensive predictions
            if not df_def.empty:
                df_def = self._ensure_feature_columns(df_def, self.defensive_features)
                preds_def = self.def_model.predict(df_def[self.defensive_features])
                df_encoded.loc[defensive_mask, 'predicted_xP_per_90'] = preds_def
                df_encoded.loc[defensive_mask, 'predicted_xP_def'] = preds_def

            # Offensive predictions + Upside
            if not df_off.empty:
                df_off = self._ensure_feature_columns(df_off, self.offensive_features)
                preds_off = self.off_model.predict(df_off[self.offensive_features])
                df_encoded.loc[offensive_mask, 'predicted_xP_per_90'] = preds_off
                df_encoded.loc[offensive_mask, 'predicted_xP_off'] = preds_off

                # Upside potential: mean + alpha * volatility_index
                volatility_series = df_off.get('volatility_index', pd.Series([0]*len(df_off)))
                upside = [
                    self._calculate_upside_potential(mu, sigma)
                    for mu, sigma in zip(preds_off, volatility_series)
                ]
                df_encoded.loc[offensive_mask, 'upside_potential'] = upside

            # 6. Validation DataFrame birleşimi
            def_val['segment'] = 'defensive'
            off_val['segment'] = 'offensive'
            validation_df = pd.concat([def_val, off_val], axis=0, ignore_index=True)

            # 7. Orijinal bilgi ekle
            df_encoded['position'] = original_df['position']
            df_encoded['web_name'] = original_df['web_name']
            df_encoded['team_name'] = original_df['team_name']
            df_encoded['price'] = original_df['price']

            # 8. Metrikler
            metrics = {
                'defensive': def_metrics,
                'offensive': off_metrics
            }

            logger.info("Segmented model workflow completed successfully")

            return df_encoded, metrics, validation_df

        except Exception as e:
            logger.error(
                f"Error in segmented train_and_predict workflow: {e}",
                exc_info=True
            )
            raise
