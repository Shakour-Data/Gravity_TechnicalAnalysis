"""
Scenario Weight Optimizer - ML-based Dynamic Weighting

این ماژول وزن‌های پویا برای سه سناریو (خوشبینانه، خنثی، بدبینانه) را
بر اساس موقعیت بازار (market regime) محاسبه می‌کند.

Author: Dr. Rajesh Kumar Patel (Algorithmic Trading Specialist)
Team: Yuki Tanaka (ML Engineer)
Date: November 14, 2025
Version: 1.0.0
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from lightgbm import LGBMRegressor
import joblib
from pathlib import Path
import structlog

logger = structlog.get_logger()


@dataclass
class MarketRegime:
    """موقعیت بازار"""
    regime_type: str  # "trending_bull", "trending_bear", "ranging", "high_volatility", "low_volatility"
    confidence: float  # 0-1
    trend_strength: float  # -1 to +1
    volatility_level: float  # 0-1
    volume_trend: float  # 0-2 (نسبت به میانگین)


@dataclass
class ScenarioWeights:
    """وزن‌های سه سناریو"""
    optimistic: float  # 0-1
    neutral: float  # 0-1
    pessimistic: float  # 0-1
    
    def __post_init__(self):
        """اطمینان از جمع = 1"""
        total = self.optimistic + self.neutral + self.pessimistic
        if not np.isclose(total, 1.0, atol=1e-6):
            # Normalize
            self.optimistic /= total
            self.neutral /= total
            self.pessimistic /= total


class ScenarioWeightOptimizer:
    """
    بهینه‌ساز وزن‌های سناریو با یادگیری ماشین
    
    این کلاس:
    1. موقعیت بازار (regime) را تشخیص می‌دهد
    2. وزن بهینه برای هر سناریو را محاسبه می‌کند
    3. از LightGBM برای یادگیری استفاده می‌کند
    
    Example:
        ```python
        optimizer = ScenarioWeightOptimizer()
        
        dimensions = {
            'trend': 0.85,
            'momentum': 0.72,
            'volatility': 0.48,
            'cycle': 0.75,
            'support_resistance': 0.63
        }
        
        weights = optimizer.calculate_weights(dimensions)
        # ScenarioWeights(optimistic=0.60, neutral=0.30, pessimistic=0.10)
        ```
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize optimizer.
        
        Args:
            model_path: مسیر مدل ذخیره شده (اختیاری)
        """
        self.model_path = model_path or Path("ml_models/scenario_weights.pkl")
        self.models: Optional[Dict[str, LGBMRegressor]] = None
        
        # وزن‌های پیش‌فرض برای هر regime
        self.default_weights = {
            'trending_bull': ScenarioWeights(0.60, 0.30, 0.10),
            'trending_bear': ScenarioWeights(0.10, 0.30, 0.60),
            'ranging': ScenarioWeights(0.25, 0.50, 0.25),
            'high_volatility': ScenarioWeights(0.20, 0.30, 0.50),
            'low_volatility': ScenarioWeights(0.30, 0.40, 0.30),
        }
        
        # Load trained model if exists
        if self.model_path.exists():
            self.load_models()
    
    def detect_regime(self, dimensions: Dict[str, float]) -> MarketRegime:
        """
        تشخیص موقعیت بازار از 5 بُعد
        
        Args:
            dimensions: {
                'trend': 0.85,
                'momentum': 0.72,
                'volatility': 0.48,
                'cycle': 0.75,
                'support_resistance': 0.63
            }
        
        Returns:
            MarketRegime object
        """
        trend = dimensions.get('trend', 0.0)
        momentum = dimensions.get('momentum', 0.0)
        volatility = dimensions.get('volatility', 0.0)
        cycle = dimensions.get('cycle', 0.0)
        sr = dimensions.get('support_resistance', 0.0)
        
        # محاسبه قدرت روند
        trend_strength = (trend + momentum) / 2.0
        
        # محاسبه سطح نوسان
        volatility_level = volatility
        
        # تشخیص regime
        regime_type = "ranging"  # default
        confidence = 0.5
        
        # 1. Trending Bull (صعودی قوی)
        if trend > 0.7 and momentum > 0.6 and volatility < 0.6:
            regime_type = "trending_bull"
            confidence = min(trend, momentum)
        
        # 2. Trending Bear (نزولی قوی)
        elif trend < -0.7 and momentum < -0.6:
            regime_type = "trending_bear"
            confidence = min(abs(trend), abs(momentum))
        
        # 3. High Volatility (نوسان بالا)
        elif volatility > 0.7:
            regime_type = "high_volatility"
            confidence = volatility
        
        # 4. Low Volatility (نوسان پایین)
        elif volatility < 0.3 and abs(trend) < 0.3:
            regime_type = "low_volatility"
            confidence = 1.0 - volatility
        
        # 5. Ranging (default)
        else:
            regime_type = "ranging"
            confidence = 1.0 - abs(trend_strength)
        
        logger.info(
            "market_regime_detected",
            regime=regime_type,
            confidence=confidence,
            trend_strength=trend_strength
        )
        
        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            volume_trend=1.0  # TODO: محاسبه از volume dimension
        )
    
    def calculate_weights(
        self,
        dimensions: Dict[str, float],
        use_ml: bool = True
    ) -> ScenarioWeights:
        """
        محاسبه وزن‌های بهینه برای سه سناریو
        
        Args:
            dimensions: امتیازات 5 بُعد
            use_ml: استفاده از ML (اگر model آموزش دیده باشد)
        
        Returns:
            ScenarioWeights با وزن‌های بهینه
        """
        # 1. تشخیص regime
        regime = self.detect_regime(dimensions)
        
        # 2. اگر ML model داریم، استفاده کنیم
        if use_ml and self.models is not None:
            weights = self._predict_weights_ml(dimensions, regime)
        else:
            # استفاده از وزن‌های پیش‌فرض
            weights = self.default_weights[regime.regime_type]
        
        logger.info(
            "scenario_weights_calculated",
            regime=regime.regime_type,
            weights={
                'optimistic': weights.optimistic,
                'neutral': weights.neutral,
                'pessimistic': weights.pessimistic
            },
            method="ml" if use_ml and self.models else "default"
        )
        
        return weights
    
    def _predict_weights_ml(
        self,
        dimensions: Dict[str, float],
        regime: MarketRegime
    ) -> ScenarioWeights:
        """
        پیش‌بینی وزن‌ها با ML model
        
        Args:
            dimensions: 5D scores
            regime: Market regime
        
        Returns:
            ScenarioWeights predicted by ML
        """
        # Feature engineering
        features = self._prepare_features(dimensions, regime)
        
        # Predict each weight
        w_opt = self.models['optimistic'].predict([features])[0]
        w_neu = self.models['neutral'].predict([features])[0]
        w_pes = self.models['pessimistic'].predict([features])[0]
        
        # Normalize to sum = 1
        weights = ScenarioWeights(w_opt, w_neu, w_pes)
        
        return weights
    
    def _prepare_features(
        self,
        dimensions: Dict[str, float],
        regime: MarketRegime
    ) -> List[float]:
        """
        آماده‌سازی ویژگی‌ها برای ML model
        
        Features (15 ویژگی):
        - 5D scores (5)
        - Regime one-hot (5)
        - Regime confidence (1)
        - Trend strength (1)
        - Volatility level (1)
        - Volume trend (1)
        - Interaction features (1)
        """
        features = []
        
        # 1. 5D scores
        features.extend([
            dimensions.get('trend', 0.0),
            dimensions.get('momentum', 0.0),
            dimensions.get('volatility', 0.0),
            dimensions.get('cycle', 0.0),
            dimensions.get('support_resistance', 0.0)
        ])
        
        # 2. Regime one-hot encoding
        regimes = ['trending_bull', 'trending_bear', 'ranging', 
                   'high_volatility', 'low_volatility']
        for r in regimes:
            features.append(1.0 if regime.regime_type == r else 0.0)
        
        # 3. Additional features
        features.extend([
            regime.confidence,
            regime.trend_strength,
            regime.volatility_level,
            regime.volume_trend
        ])
        
        # 4. Interaction feature (trend × volatility)
        features.append(regime.trend_strength * regime.volatility_level)
        
        return features
    
    def train(
        self,
        training_data: List[Dict],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        آموزش ML models
        
        Args:
            training_data: لیست از نمونه‌های آموزشی
                [
                    {
                        'dimensions': {...},
                        'regime': MarketRegime(...),
                        'actual_best_weights': ScenarioWeights(...)
                    },
                    ...
                ]
            validation_split: درصد داده برای validation
        
        Returns:
            metrics: {
                'optimistic_mae': 0.05,
                'neutral_mae': 0.04,
                'pessimistic_mae': 0.06
            }
        """
        logger.info("training_scenario_weight_models", samples=len(training_data))
        
        # Prepare X and y
        X = []
        y_opt = []
        y_neu = []
        y_pes = []
        
        for sample in training_data:
            features = self._prepare_features(
                sample['dimensions'],
                sample['regime']
            )
            X.append(features)
            
            weights = sample['actual_best_weights']
            y_opt.append(weights.optimistic)
            y_neu.append(weights.neutral)
            y_pes.append(weights.pessimistic)
        
        X = np.array(X)
        y_opt = np.array(y_opt)
        y_neu = np.array(y_neu)
        y_pes = np.array(y_pes)
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_opt_train, y_opt_val = y_opt[:split_idx], y_opt[split_idx:]
        y_neu_train, y_neu_val = y_neu[:split_idx], y_neu[split_idx:]
        y_pes_train, y_pes_val = y_pes[:split_idx], y_pes[split_idx:]
        
        # Train 3 separate models
        self.models = {}
        metrics = {}
        
        for target_name, y_train, y_val in [
            ('optimistic', y_opt_train, y_opt_val),
            ('neutral', y_neu_train, y_neu_val),
            ('pessimistic', y_pes_train, y_pes_val)
        ]:
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            # Validation
            y_pred = model.predict(X_val)
            mae = np.mean(np.abs(y_pred - y_val))
            
            self.models[target_name] = model
            metrics[f'{target_name}_mae'] = mae
            
            logger.info(
                f"model_trained_{target_name}",
                mae=mae,
                train_samples=len(X_train),
                val_samples=len(X_val)
            )
        
        # Save models
        self.save_models()
        
        return metrics
    
    def save_models(self):
        """ذخیره ML models"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, self.model_path)
        logger.info("models_saved", path=str(self.model_path))
    
    def load_models(self):
        """بارگذاری ML models"""
        if self.model_path.exists():
            self.models = joblib.load(self.model_path)
            logger.info("models_loaded", path=str(self.model_path))
        else:
            logger.warning("no_saved_models_found", path=str(self.model_path))


# Singleton instance
_optimizer_instance: Optional[ScenarioWeightOptimizer] = None


def get_scenario_weight_optimizer() -> ScenarioWeightOptimizer:
    """Get singleton instance of ScenarioWeightOptimizer"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ScenarioWeightOptimizer()
    return _optimizer_instance
