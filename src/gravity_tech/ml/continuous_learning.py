"""
Continuous Learning System - Learning from Experience

This system:
1. Learns from its mistakes (incorrect predictions)
2. Uses experiences across different symbols
3. Automatically updates models
4. Tracks performance in real-time

This module provides a comprehensive continuous learning framework that:
- Records all predictions and actual outcomes
- Calculates prediction errors and learns from them
- Maintains separate performance profiles for each symbol
- Automatically retrains models every N predictions
- Persists learning history to disk for long-term memory
- Provides insights into symbol-specific and cross-symbol patterns

The learning system enables the technical analysis to improve over time by:
- Adjusting weights based on historical accuracy
- Identifying which market phases work best for each symbol
- Discovering patterns that work across multiple symbols
- Continuously refining prediction models with new data

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from gravity_tech.ml.weight_optimizer import MLWeightOptimizer

logger = structlog.get_logger()


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    timestamp: datetime
    symbol: str
    timeframe: str
    predicted_signal: float  # -10 to +10
    actual_return: Optional[float]  # Actual return percentage
    prediction_error: Optional[float]  # Prediction error
    market_phase: str
    weights_used: dict[str, float]
    indicators_used: dict[str, float]
    confidence: float


@dataclass
class SymbolPerformance:
    """Performance metrics for a symbol"""
    symbol: str
    total_predictions: int
    correct_predictions: int  # Correct direction
    accuracy: float  # Accuracy percentage
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    last_update: datetime


class ContinuousLearner:
    """
    Continuous Learning System

    Features:
    - Stores all predictions
    - Calculates error after each prediction
    - Automatically updates model
    - Learns from different symbols

    Example:
        ```python
        learner = ContinuousLearner()

        # Record prediction
        learner.record_prediction(
            symbol="BTCUSDT",
            predicted_signal=7.5,
            market_phase="uptrend",
            weights={'trend': 0.4, 'momentum': 0.3},
            indicators={'sma': 8.2, 'rsi': 65}
        )

        # After 1 hour, update with actual result
        learner.update_actual_result(
            symbol="BTCUSDT",
            actual_return=5.2  # Price went up 5.2%
        )

        # Learn from mistakes
        learner.learn_from_mistakes()
        ```
    """

    def __init__(
        self,
        db_path: Path = Path("data/learning_history.db"),
        retrain_interval: int = 100,  # Retrain model every 100 predictions
        max_history: int = 10000
    ):
        """
        Initialize continuous learner

        Args:
            db_path: Path to history database
            retrain_interval: Number of new predictions before retraining
            max_history: Maximum number of records to keep
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.retrain_interval = retrain_interval
        self.max_history = max_history

        # Prediction history (in memory)
        self.prediction_history: deque = deque(maxlen=max_history)

        # Performance per symbol
        self.symbol_performance: dict[str, SymbolPerformance] = {}

        # ML model
        self.ml_optimizer = MLWeightOptimizer()

        # Counter for retraining
        self.predictions_since_retrain = 0

        # Load previous history
        self._load_history()

        logger.info(
            "continuous_learner_initialized",
            db_path=str(db_path),
            history_size=len(self.prediction_history),
            symbols=list(self.symbol_performance.keys())
        )

    def record_prediction(
        self,
        symbol: str,
        timeframe: str,
        predicted_signal: float,
        market_phase: str,
        weights_used: dict[str, float],
        indicators_used: dict[str, float],
        confidence: float = 0.5
    ) -> str:
        """
        Record a new prediction

        Args:
            symbol: Symbol (e.g., BTCUSDT)
            timeframe: Timeframe (1h, 4h, 1d)
            predicted_signal: Predicted signal (-10 to +10)
            market_phase: Market phase
            weights_used: Weights used
            indicators_used: Indicator values
            confidence: Prediction confidence

        Returns:
            prediction_id: Unique identifier for prediction
        """
        record = PredictionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            predicted_signal=predicted_signal,
            actual_return=None,  # Not yet known
            prediction_error=None,
            market_phase=market_phase,
            weights_used=weights_used,
            indicators_used=indicators_used,
            confidence=confidence
        )

        self.prediction_history.append(record)

        prediction_id = f"{symbol}_{int(record.timestamp.timestamp())}"

        logger.info(
            "prediction_recorded",
            symbol=symbol,
            predicted_signal=predicted_signal,
            confidence=confidence,
            prediction_id=prediction_id
        )

        return prediction_id

    def update_actual_result(
        self,
        symbol: str,
        actual_return: float,
        timestamp: Optional[datetime] = None,
        max_age_hours: int = 24
    ) -> bool:
        """
        Update actual result of a prediction

        Args:
            symbol: Symbol
            actual_return: Actual return (percentage)
            timestamp: Prediction timestamp (if None, uses latest prediction)
            max_age_hours: Maximum age of prediction for update

        Returns:
            True if update was successful
        """
        # Find related prediction
        target_record = None

        if timestamp:
            # Search by timestamp
            for record in reversed(self.prediction_history):
                if (record.symbol == symbol and
                    abs((record.timestamp - timestamp).total_seconds()) < 60):
                    target_record = record
                    break
        else:
            # Latest prediction for this symbol
            for record in reversed(self.prediction_history):
                if record.symbol == symbol and record.actual_return is None:
                    age_hours = (datetime.now() - record.timestamp).total_seconds() / 3600
                    if age_hours <= max_age_hours:
                        target_record = record
                        break

        if not target_record:
            logger.warning(
                "prediction_not_found_for_update",
                symbol=symbol,
                timestamp=timestamp
            )
            return False

        # Calculate error
        predicted_direction = 1 if target_record.predicted_signal > 0 else -1
        actual_direction = 1 if actual_return > 0 else -1

        # Absolute error
        error = abs(target_record.predicted_signal - actual_return)

        # Update record
        target_record.actual_return = actual_return
        target_record.prediction_error = error

        # Update symbol statistics
        self._update_symbol_performance(symbol, predicted_direction, actual_direction, error)

        # Count for retraining
        self.predictions_since_retrain += 1

        logger.info(
            "actual_result_updated",
            symbol=symbol,
            predicted=target_record.predicted_signal,
            actual=actual_return,
            error=error,
            direction_correct=(predicted_direction == actual_direction)
        )

        # Retrain if necessary
        if self.predictions_since_retrain >= self.retrain_interval:
            asyncio.create_task(self.retrain_model())

        return True

    def _update_symbol_performance(
        self,
        symbol: str,
        predicted_direction: int,
        actual_direction: int,
        error: float
    ):
        """Update symbol performance metrics"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = SymbolPerformance(
                symbol=symbol,
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                mae=0.0,
                rmse=0.0,
                last_update=datetime.now()
            )

        perf = self.symbol_performance[symbol]
        perf.total_predictions += 1

        if predicted_direction == actual_direction:
            perf.correct_predictions += 1

        perf.accuracy = perf.correct_predictions / perf.total_predictions

        # Calculate new MAE and RMSE (moving average)
        alpha = 0.1  # Weight coefficient
        perf.mae = (1 - alpha) * perf.mae + alpha * error
        perf.rmse = np.sqrt((1 - alpha) * (perf.rmse ** 2) + alpha * (error ** 2))

        perf.last_update = datetime.now()

    async def retrain_model(self):
        """Retrain model with new experiences"""
        logger.info("retrain_model_started", predictions_count=self.predictions_since_retrain)

        # Collect training data from history
        training_data = []

        for record in self.prediction_history:
            if record.actual_return is not None:  # Only records with actual results
                # Build features
                features = {
                    **record.weights_used,
                    **record.indicators_used,
                    'confidence': record.confidence,
                    'market_phase_encoded': self._encode_market_phase(record.market_phase)
                }

                training_data.append({
                    'features': features,
                    'target': record.actual_return,
                    'symbol': record.symbol
                })

        if len(training_data) < 50:
            logger.warning("insufficient_data_for_retrain", count=len(training_data))
            return

        try:
            # Train model
            metrics = self.ml_optimizer.train(training_data, validation_split=0.2)

            # Save model
            self.ml_optimizer.save_model(name=f"continuous_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            self.predictions_since_retrain = 0

            logger.info(
                "retrain_model_completed",
                training_samples=len(training_data),
                validation_r2=metrics.get('val_r2'),
                symbols=len(set(d['symbol'] for d in training_data))
            )

        except Exception as e:
            logger.error("retrain_model_failed", error=str(e))

    def get_symbol_insights(self, symbol: str) -> Optional[dict]:
        """
        Get learning insights for a symbol

        Returns:
            Insights including: accuracy, best timeframe, best market phase
        """
        if symbol not in self.symbol_performance:
            return None

        perf = self.symbol_performance[symbol]

        # Analyze history for this symbol
        symbol_records = [r for r in self.prediction_history if r.symbol == symbol and r.actual_return is not None]

        if not symbol_records:
            return None

        # Best timeframe
        timeframe_performance = {}
        for record in symbol_records:
            tf = record.timeframe
            if tf not in timeframe_performance:
                timeframe_performance[tf] = {'correct': 0, 'total': 0}

            timeframe_performance[tf]['total'] += 1
            if (record.predicted_signal > 0 and record.actual_return > 0) or \
               (record.predicted_signal < 0 and record.actual_return < 0):
                timeframe_performance[tf]['correct'] += 1

        best_timeframe = max(
            timeframe_performance.items(),
            key=lambda x: x[1]['correct'] / x[1]['total']
        )[0] if timeframe_performance else None

        # Best market phase
        phase_performance = {}
        for record in symbol_records:
            phase = record.market_phase
            if phase not in phase_performance:
                phase_performance[phase] = {'correct': 0, 'total': 0}

            phase_performance[phase]['total'] += 1
            if (record.predicted_signal > 0 and record.actual_return > 0) or \
               (record.predicted_signal < 0 and record.actual_return < 0):
                phase_performance[phase]['correct'] += 1

        best_phase = max(
            phase_performance.items(),
            key=lambda x: x[1]['correct'] / x[1]['total']
        )[0] if phase_performance else None

        return {
            'symbol': symbol,
            'overall_accuracy': perf.accuracy,
            'total_predictions': perf.total_predictions,
            'mae': perf.mae,
            'rmse': perf.rmse,
            'best_timeframe': best_timeframe,
            'best_market_phase': best_phase,
            'last_update': perf.last_update.isoformat(),
            'timeframe_performance': {
                tf: {
                    'accuracy': data['correct'] / data['total'],
                    'total': data['total']
                }
                for tf, data in timeframe_performance.items()
            },
            'phase_performance': {
                phase: {
                    'accuracy': data['correct'] / data['total'],
                    'total': data['total']
                }
                for phase, data in phase_performance.items()
            }
        }

    def get_cross_symbol_patterns(self) -> dict[str, any]:
        """
        Cross-symbol patterns analysis

        Returns:
            Patterns that work across all symbols
        """
        # Analyze performance across different market phases
        global_phase_performance = {}

        for record in self.prediction_history:
            if record.actual_return is None:
                continue

            phase = record.market_phase
            if phase not in global_phase_performance:
                global_phase_performance[phase] = {'correct': 0, 'total': 0, 'avg_error': []}

            global_phase_performance[phase]['total'] += 1

            if (record.predicted_signal > 0 and record.actual_return > 0) or \
               (record.predicted_signal < 0 and record.actual_return < 0):
                global_phase_performance[phase]['correct'] += 1

            global_phase_performance[phase]['avg_error'].append(record.prediction_error)

        # Calculate final statistics
        for phase, data in global_phase_performance.items():
            data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0
            data['avg_error'] = np.mean(data['avg_error']) if data['avg_error'] else 0
            del data['avg_error']  # Remove raw list

        return {
            'phase_performance': global_phase_performance,
            'total_symbols': len(self.symbol_performance),
            'total_predictions': len(self.prediction_history),
            'best_performing_symbols': sorted(
                self.symbol_performance.values(),
                key=lambda x: x.accuracy,
                reverse=True
            )[:5]
        }

    def _encode_market_phase(self, phase: str) -> float:
        """Encode market phase to numerical value"""
        encoding = {
            'accumulation': 0.0,
            'uptrend': 0.25,
            'distribution': 0.5,
            'downtrend': 0.75,
            'transition': 0.5
        }
        return encoding.get(phase, 0.5)

    def _load_history(self):
        """Load history from disk"""
        history_file = self.db_path.parent / "prediction_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file, encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct records
            for record_dict in data.get('predictions', []):
                record = PredictionRecord(
                    timestamp=datetime.fromisoformat(record_dict['timestamp']),
                    symbol=record_dict['symbol'],
                    timeframe=record_dict['timeframe'],
                    predicted_signal=record_dict['predicted_signal'],
                    actual_return=record_dict.get('actual_return'),
                    prediction_error=record_dict.get('prediction_error'),
                    market_phase=record_dict['market_phase'],
                    weights_used=record_dict['weights_used'],
                    indicators_used=record_dict['indicators_used'],
                    confidence=record_dict['confidence']
                )
                self.prediction_history.append(record)

            # Reconstruct symbol performance
            for symbol, perf_dict in data.get('symbol_performance', {}).items():
                self.symbol_performance[symbol] = SymbolPerformance(
                    symbol=symbol,
                    total_predictions=perf_dict['total_predictions'],
                    correct_predictions=perf_dict['correct_predictions'],
                    accuracy=perf_dict['accuracy'],
                    mae=perf_dict['mae'],
                    rmse=perf_dict['rmse'],
                    last_update=datetime.fromisoformat(perf_dict['last_update'])
                )

            logger.info(
                "history_loaded",
                predictions=len(self.prediction_history),
                symbols=len(self.symbol_performance)
            )

        except Exception as e:
            logger.error("history_load_failed", error=str(e))

    def save_history(self):
        """Save history to disk"""
        history_file = self.db_path.parent / "prediction_history.json"

        try:
            data = {
                'predictions': [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'symbol': record.symbol,
                        'timeframe': record.timeframe,
                        'predicted_signal': record.predicted_signal,
                        'actual_return': record.actual_return,
                        'prediction_error': record.prediction_error,
                        'market_phase': record.market_phase,
                        'weights_used': record.weights_used,
                        'indicators_used': record.indicators_used,
                        'confidence': record.confidence
                    }
                    for record in self.prediction_history
                ],
                'symbol_performance': {
                    symbol: {
                        'total_predictions': perf.total_predictions,
                        'correct_predictions': perf.correct_predictions,
                        'accuracy': perf.accuracy,
                        'mae': perf.mae,
                        'rmse': perf.rmse,
                        'last_update': perf.last_update.isoformat()
                    }
                    for symbol, perf in self.symbol_performance.items()
                }
            }

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info("history_saved", file=str(history_file))

        except Exception as e:
            logger.error("history_save_failed", error=str(e))


# Global instance
continuous_learner = ContinuousLearner()
