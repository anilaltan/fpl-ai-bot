"""
News Radar Module - Sakatlık ve Risk Yönetimi

Bu modül, oyuncuların oynama ihtimallerini analiz ederek availability skorlarını hesaplar.
Sakatlık haberlerini işler ve risk faktörlerini değerlendirir.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional

logger = logging.getLogger(__name__)


class NewsRadar:
    """
    FPL oyuncularının availability durumunu analiz eden sınıf.

    Sakatlık haberlerini işler, oynama ihtimallerini değerlendirir ve
    risk-adjusted puan tahminleri üretir.
    """

    def __init__(self):
        """Initialize NewsRadar with default risk parameters."""
        self.illness_boost_threshold = 75.0  # %75 üzerindeki hastalık vakaları için boost
        self.illness_boost_factor = 0.85     # Hastalık boost faktörü
        self.doubtful_risk_penalty = 0.5     # Doubtful durumundaki oyuncular için ceza
        self.injured_penalty = 0.0           # Injured durumundaki oyuncular için ceza
        self.suspended_penalty = 0.0         # Suspended durumundaki oyuncular için ceza

    def calculate_availability_score(self, row: pd.Series) -> float:
        """
        Oyuncunun oynama ihtimalini hesaplar ve availability skorunu döner.

        Args:
            row: Oyuncu verilerini içeren pandas Series

        Returns:
            float: 0.0-1.0 arası availability skoru (1.0 = tam müsait)
        """
        try:
            # Ana oynama ihtimali kontrolü
            chance_next = row.get('chance_of_playing_next_round')
            status = row.get('status', '').lower() if pd.notna(row.get('status')) else None
            news = str(row.get('news', '')).lower() if pd.notna(row.get('news')) else ''

            # 1. Öncelik: chance_of_playing_next_round varsa onu kullan
            if pd.notna(chance_next) and chance_next is not None:
                base_score = float(chance_next) / 100.0

                # Hastalık boost kontrolü
                if self._has_illness_boost(news, chance_next):
                    base_score = min(1.0, base_score * 1.15)  # Hafif boost

                return self._apply_text_analysis(base_score, news)

            # 2. Fallback: status bilgisine göre karar ver
            if status:
                if status == 'a':  # Available
                    base_score = 1.0
                elif status == 's':  # Suspended
                    base_score = self.suspended_penalty
                elif status == 'i':  # Injured
                    base_score = self.injured_penalty
                elif status == 'd':  # Doubtful
                    base_score = self.doubtful_risk_penalty
                else:
                    # Bilinmeyen status için conservative yaklaşım
                    base_score = 0.5

                return self._apply_text_analysis(base_score, news)

            # 3. Hiç veri yoksa tam müsait kabul et
            logger.debug(f"Oyuncu {row.get('web_name', 'Unknown')} için availability verisi bulunamadı, 1.0 varsayılıyor")
            return 1.0

        except Exception as e:
            logger.warning(f"Availability score hesaplama hatası: {e}")
            return 1.0  # Hata durumunda tam müsait kabul et

    def _has_illness_boost(self, news: str, chance: float) -> bool:
        """
        Hastalık vakalarında boost uygulanıp uygulanmayacağını kontrol eder.

        Args:
            news: Sakatlık haberi metni
            chance: Oynama ihtimali yüzdesi

        Returns:
            bool: Boost uygulanıp uygulanmayacağı
        """
        if 'illness' in news and chance >= self.illness_boost_threshold:
            return True
        return False

    def _apply_text_analysis(self, base_score: float, news: str) -> float:
        """
        Haber metnindeki risk faktörlerini analiz eder ve skoru ayarlar.

        Args:
            base_score: Temel availability skoru
            news: Sakatlık haberi metni

        Returns:
            float: Ayarlanmış availability skoru
        """
        adjusted_score = base_score

        # Riskli sakatlıklar için ceza uygula
        if 'hamstring' in news:
            adjusted_score *= 0.7  # Hamstring ciddi risk
            logger.debug("Hamstring cezası uygulandı")
        elif 'knee' in news:
            adjusted_score *= 0.6  # Knee ciddi risk
            logger.debug("Knee cezası uygulandı")
        elif 'ankle' in news:
            adjusted_score *= 0.8  # Ankle orta risk
        elif 'groin' in news:
            adjusted_score *= 0.75  # Groin ciddi risk

        # Çok uzun süreli sakatlıklar için ek ceza
        if any(term in news for term in ['expected back', 'return date']):
            try:
                # Basit tarih parsing - gerçek implementasyonda daha sofistike olabilir
                if 'jan' in news or 'feb' in news or 'mar' in news:
                    adjusted_score *= 0.5  # Çok uzun süre
                elif 'week' in news and ('3' in news or '4' in news):
                    adjusted_score *= 0.7  # 3-4 hafta
            except:
                pass  # Parsing hatası olursa devam et

        # Sınır kontrolü
        return np.clip(adjusted_score, 0.0, 1.0)

    def analyze_player_risk(self, row: pd.Series) -> dict:
        """
        Oyuncunun detaylı risk analizini döner.

        Args:
            row: Oyuncu verilerini içeren pandas Series

        Returns:
            dict: Risk analiz detayları
        """
        availability_score = self.calculate_availability_score(row)

        risk_level = self._categorize_risk(availability_score)

        return {
            'availability_score': availability_score,
            'risk_level': risk_level,
            'chance_next_round': row.get('chance_of_playing_next_round'),
            'chance_this_round': row.get('chance_of_playing_this_round'),
            'status': row.get('status'),
            'news': row.get('news', ''),
            'is_high_risk': availability_score < 0.75,
            'recommendation': self._get_risk_recommendation(availability_score, row)
        }

    def _categorize_risk(self, score: float) -> str:
        """Risk seviyesini kategorize eder."""
        if score >= 0.9:
            return 'LOW'
        elif score >= 0.75:
            return 'MEDIUM'
        elif score >= 0.5:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def _get_risk_recommendation(self, score: float, row: pd.Series) -> str:
        """Risk durumuna göre öneri üretir."""
        if score >= 0.9:
            return "✅ Güvenli - Transfer edilebilir"
        elif score >= 0.75:
            return "⚠️ Dikkat - Riskli olabilir"
        elif score >= 0.5:
            return "❌ Yüksek Risk - Transfer önerilmez"
        else:
            return "🚫 Kritik Risk - Acil transfer gerekli"


def calculate_availability_score(row: pd.Series) -> float:
    """
    Convenience function for direct availability score calculation.

    Args:
        row: Oyuncu verilerini içeren pandas Series

    Returns:
        float: 0.0-1.0 arası availability skoru
    """
    radar = NewsRadar()
    return radar.calculate_availability_score(row)


def analyze_player_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm oyuncular için risk analizini uygular.

    Args:
        df: Oyuncu verilerini içeren DataFrame

    Returns:
        pd.DataFrame: Risk analiz sütunları eklenmiş DataFrame
    """
    radar = NewsRadar()

    # Availability skorlarını hesapla
    df = df.copy()
    df['availability_score'] = df.apply(radar.calculate_availability_score, axis=1)

    # Risk analizlerini ekle
    risk_analyses = []
    for _, row in df.iterrows():
        analysis = radar.analyze_player_risk(row)
        risk_analyses.append(analysis)

    # Analiz sonuçlarını DataFrame'e ekle
    risk_df = pd.DataFrame(risk_analyses)
    result_df = pd.concat([df.reset_index(drop=True), risk_df], axis=1)

    # Log summary
    risk_counts = result_df['risk_level'].value_counts()
    logger.info(f"Risk analizi tamamlandı: {dict(risk_counts)}")

    return result_df
