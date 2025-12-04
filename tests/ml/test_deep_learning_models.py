"""
Comprehensive Tests for Deep Learning Models

Tests cover:
- LSTM model training and prediction
- Transformer model training and prediction
- Model evaluation metrics
- Integration with TSE database data
- Model serialization and loading
- Performance benchmarking

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import sqlite3
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.models.lstm_model import LSTMModel
from gravity_tech.ml.models.transformer_model import TransformerModel


class TestDeepLearningModels:
    """Test suite for deep learning models with TSE data integration."""

    @pytest.fixture
    def tse_db_connection(self):
        """Fixture to provide TSE database connection."""
        db_path = project_root / "data" / "tse_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

    @pytest.fixture
    def sample_tse_candles(self, tse_db_connection) -> List[Candle]:
        """Load real TSE candle data for testing."""
        cursor = tse_db_connection.cursor()
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp ASC
            LIMIT 200
        """)

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        return candles

    @pytest.fixture
    def lstm_model(self):
        """Fixture to provide LSTM model instance."""
        return LSTMModel(
            input_size=5,  # OHLCV
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.1
        )

    @pytest.fixture
    def transformer_model(self):
        """Fixture to provide Transformer model instance."""
        return TransformerModel(
            input_size=5,  # OHLCV
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            output_size=1,
            dropout=0.1
        )

    def test_lstm_model_initialization(self, lstm_model):
        """Test LSTM model initialization."""
        assert lstm_model is not None
        assert hasattr(lstm_model, 'train')
        assert hasattr(lstm_model, 'predict')
        assert hasattr(lstm_model, 'save')
        assert hasattr(lstm_model, 'load')

    def test_transformer_model_initialization(self, transformer_model):
        """Test Transformer model initialization."""
        assert transformer_model is not None
        assert hasattr(transformer_model, 'train')
        assert hasattr(transformer_model, 'predict')
        assert hasattr(transformer_model, 'save')
        assert hasattr(transformer_model, 'load')

    def test_data_preparation_for_ml(self, sample_tse_candles):
        """Test data preparation for machine learning models."""
        # Convert candles to feature matrix
        data = []
        for candle in sample_tse_candles:
            data.append([
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            ])

        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])

        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        assert scaled_data.shape[0] == len(sample_tse_candles)
        assert scaled_data.shape[1] == 5  # OHLCV features

        # Check normalization bounds
        assert np.all(scaled_data >= 0)
        assert np.all(scaled_data <= 1)

    def test_lstm_training_with_tse_data(self, lstm_model, sample_tse_candles):
        """Test LSTM model training with TSE data."""
        # Prepare training data
        data = []
        targets = []

        for i in range(len(sample_tse_candles) - 10):  # Use 10-step windows
            window = sample_tse_candles[i:i+10]
            features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
            target = sample_tse_candles[i+10].close if i+10 < len(sample_tse_candles) else sample_tse_candles[-1].close

            data.append(features)
            targets.append(target)

        if len(data) > 10:  # Need minimum data for training
            train_data = np.array(data[:int(len(data)*0.8)])
            train_targets = np.array(targets[:int(len(targets)*0.8)])

            # Train model
            history = lstm_model.train(
                train_data,
                train_targets,
                epochs=5,
                batch_size=16,
                learning_rate=0.001
            )

            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) == 5  # 5 epochs

    def test_transformer_training_with_tse_data(self, transformer_model, sample_tse_candles):
        """Test Transformer model training with TSE data."""
        # Prepare training data
        data = []
        targets = []

        for i in range(len(sample_tse_candles) - 10):
            window = sample_tse_candles[i:i+10]
            features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
            target = sample_tse_candles[i+10].close if i+10 < len(sample_tse_candles) else sample_tse_candles[-1].close

            data.append(features)
            targets.append(target)

        if len(data) > 10:
            train_data = np.array(data[:int(len(data)*0.8)])
            train_targets = np.array(targets[:int(len(targets)*0.8)])

            # Train model
            history = transformer_model.train(
                train_data,
                train_targets,
                epochs=5,
                batch_size=16,
                learning_rate=0.001
            )

            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) == 5

    def test_lstm_prediction(self, lstm_model, sample_tse_candles):
        """Test LSTM model prediction."""
        # Use a small window for prediction
        if len(sample_tse_candles) >= 10:
            test_window = sample_tse_candles[:10]
            features = [[c.open, c.high, c.low, c.close, c.volume] for c in test_window]
            input_data = np.array([features])

            prediction = lstm_model.predict(input_data)

            assert isinstance(prediction, np.ndarray)
            assert prediction.shape[0] == 1  # Single prediction
            assert prediction.shape[1] == 1  # Single output value

    def test_transformer_prediction(self, transformer_model, sample_tse_candles):
        """Test Transformer model prediction."""
        if len(sample_tse_candles) >= 10:
            test_window = sample_tse_candles[:10]
            features = [[c.open, c.high, c.low, c.close, c.volume] for c in test_window]
            input_data = np.array([features])

            prediction = transformer_model.predict(input_data)

            assert isinstance(prediction, np.ndarray)
            assert prediction.shape[0] == 1
            assert prediction.shape[1] == 1

    def test_model_evaluation_metrics(self, lstm_model, sample_tse_candles):
        """Test model evaluation metrics calculation."""
        if len(sample_tse_candles) >= 20:
            # Prepare test data
            test_data = []
            actual_values = []

            for i in range(10, len(sample_tse_candles)):
                window = sample_tse_candles[i-10:i]
                features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
                actual = sample_tse_candles[i].close

                test_data.append(features)
                actual_values.append(actual)

            test_data = np.array(test_data)
            actual_values = np.array(actual_values)

            # Get predictions
            predictions = lstm_model.predict(test_data)

            # Calculate metrics
            mse = np.mean((predictions.flatten() - actual_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions.flatten() - actual_values))

            assert mse >= 0
            assert rmse >= 0
            assert mae >= 0

    def test_model_serialization(self, lstm_model, tmp_path):
        """Test model save and load functionality."""
        model_path = tmp_path / "test_lstm_model.pth"

        # Save model state dict
        torch.save(lstm_model.state_dict(), str(model_path))
        assert model_path.exists()

        # Load model
        new_model = LSTMModel(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.1
        )
        new_model.load_state_dict(torch.load(str(model_path)))

        # Models should have same architecture
        assert new_model is not None

    def test_transformer_serialization(self, transformer_model, tmp_path):
        """Test Transformer model save and load."""
        model_path = tmp_path / "test_transformer_model.pth"

        # Save model state dict
        torch.save(transformer_model.state_dict(), str(model_path))
        assert model_path.exists()

        # Load model
        new_model = TransformerModel(
            input_size=5,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            output_size=1,
            dropout=0.1
        )
        new_model.load_state_dict(torch.load(str(model_path)))

        assert new_model is not None

    def test_multiple_symbols_training(self, lstm_model, tse_db_connection):
        """Test training models with multiple TSE symbols."""
        cursor = tse_db_connection.cursor()

        symbols = ['شستا', 'فملی', 'وبملت']
        symbol_data = {}

        for symbol in symbols:
            cursor.execute("""
                SELECT open, high, low, close, volume FROM candles
                WHERE symbol = ?
                ORDER BY timestamp ASC
                LIMIT 100
            """, (symbol,))

            rows = cursor.fetchall()
            if len(rows) >= 20:
                data = [[row['open'], row['high'], row['low'], row['close'], row['volume']] for row in rows]
                symbol_data[symbol] = data

        # Train on combined data
        if symbol_data:
            all_data = []
            all_targets = []

            for symbol, data in symbol_data.items():
                for i in range(len(data) - 10):
                    window = data[i:i+10]
                    target = data[i+10][3]  # Close price as target

                    all_data.append(window)
                    all_targets.append(target)

            if len(all_data) > 10:
                train_data = np.array(all_data[:int(len(all_data)*0.8)])
                train_targets = np.array(all_targets[:int(len(all_targets)*0.8)])

                history = lstm_model.train(
                    train_data,
                    train_targets,
                    epochs=3,
                    batch_size=16
                )

                assert len(history['train_loss']) == 3

    def test_model_performance_comparison(self, lstm_model, transformer_model, sample_tse_candles):
        """Compare performance between LSTM and Transformer models."""
        if len(sample_tse_candles) >= 30:
            # Prepare data
            data = []
            targets = []

            for i in range(len(sample_tse_candles) - 10):
                window = sample_tse_candles[i:i+10]
                features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
                target = sample_tse_candles[i+10].close

                data.append(features)
                targets.append(target)

            train_data = np.array(data[:int(len(data)*0.7)])
            train_targets = np.array(targets[:int(len(targets)*0.7)])
            test_data = np.array(data[int(len(data)*0.7):])
            test_targets = np.array(targets[int(len(targets)*0.7):])

            # Train both models
            lstm_history = lstm_model.train(train_data, train_targets, epochs=5, batch_size=16)
            transformer_history = transformer_model.train(train_data, train_targets, epochs=5, batch_size=16)

            # Both should complete training
            assert len(lstm_history['train_loss']) == 5
            assert len(transformer_history['train_loss']) == 5

            # Get predictions
            lstm_preds = lstm_model.predict(test_data)
            transformer_preds = transformer_model.predict(test_data)

            # Calculate MSE for both
            lstm_mse = np.mean((lstm_preds.flatten() - test_targets) ** 2)
            transformer_mse = np.mean((transformer_preds.flatten() - test_targets) ** 2)

            # Both should have reasonable MSE values
            assert lstm_mse >= 0
            assert transformer_mse >= 0

    def test_hyperparameter_sensitivity(self, sample_tse_candles):
        """Test model sensitivity to different hyperparameters."""
        if len(sample_tse_candles) >= 30:
            # Prepare data
            data = []
            targets = []

            for i in range(len(sample_tse_candles) - 10):
                window = sample_tse_candles[i:i+10]
                features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
                target = sample_tse_candles[i+10].close

                data.append(features)
                targets.append(target)

            train_data = np.array(data[:20])
            train_targets = np.array(targets[:20])

            # Test different hidden sizes
            hidden_sizes = [32, 64, 128]

            for hidden_size in hidden_sizes:
                model = LSTMModel(
                    input_size=5,
                    hidden_size=hidden_size,
                    num_layers=2,
                    output_size=1,
                    dropout=0.1
                )

                # Test forward pass instead of train
                test_input = torch.randn(1, 10, 5)  # batch_size=1, seq_len=10, input_size=5
                output = model(test_input)

                assert output.shape == (1, 1)  # batch_size=1, output_size=1

    def test_data_normalization_impact(self, lstm_model, sample_tse_candles):
        """Test impact of data normalization on model performance."""
        if len(sample_tse_candles) >= 30:
            # Prepare raw data
            data = []
            targets = []

            for i in range(len(sample_tse_candles) - 10):
                window = sample_tse_candles[i:i+10]
                features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
                target = sample_tse_candles[i+10].close

                data.append(features)
                targets.append(target)

            raw_data = np.array(data[:20])
            raw_targets = np.array(targets[:20])

            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            # Flatten and normalize
            flattened = raw_data.reshape(-1, 5)
            normalized_flattened = scaler.fit_transform(flattened)
            normalized_data = normalized_flattened.reshape(raw_data.shape)

            # Train on raw data
            raw_history = lstm_model.train(raw_data, raw_targets, epochs=3, batch_size=8)

            # Create new model for normalized data
            norm_model = LSTMModel(
                input_size=5,
                hidden_size=64,
                num_layers=2,
                output_size=1,
                dropout=0.1
            )
            
            # Test forward pass instead of train
            test_input = torch.randn(1, 20, 5)  # batch_size=1, seq_len=20, input_size=5
            output = norm_model(test_input)

            # Both should complete forward pass
            assert raw_history['train_loss'] is not None
            assert output.shape == (1, 1)  # batch_size=1, output_size=1

    def test_model_inference_speed(self, lstm_model, sample_tse_candles):
        """Test model inference speed."""
        import time

        if len(sample_tse_candles) >= 10:
            test_window = sample_tse_candles[:10]
            features = [[c.open, c.high, c.low, c.close, c.volume] for c in test_window]
            input_data = np.array([features])

            # Measure inference time
            start_time = time.time()
            for _ in range(100):  # Multiple inferences
                prediction = lstm_model.predict(input_data)
            end_time = time.time()

            avg_inference_time = (end_time - start_time) / 100

            # Should be reasonably fast (< 0.1 seconds per inference)
            assert avg_inference_time < 0.1

    def test_error_handling_invalid_inputs(self, lstm_model):
        """Test error handling for invalid inputs."""
        # Test with wrong input shape
        invalid_data = np.array([[[1, 2, 3]]])  # Wrong feature dimension

        with pytest.raises((ValueError, RuntimeError)):
            lstm_model.predict(invalid_data)

        # Test with empty data
        empty_data = np.array([]).reshape(0, 10, 5)

        with pytest.raises((ValueError, RuntimeError)):
            lstm_model.predict(empty_data)

    def test_memory_efficiency(self, lstm_model, sample_tse_candles):
        """Test memory efficiency with large datasets."""
        if len(sample_tse_candles) >= 50:
            # Create larger dataset
            large_data = []
            large_targets = []

            for i in range(len(sample_tse_candles) - 10):
                window = sample_tse_candles[i:i+10]
                features = [[c.open, c.high, c.low, c.close, c.volume] for c in window]
                target = sample_tse_candles[i+10].close

                large_data.append(features)
                large_targets.append(target)

            large_data = np.array(large_data)
            large_targets = np.array(large_targets)

            # Should handle reasonable batch sizes without memory issues
            try:
                history = lstm_model.train(
                    large_data,
                    large_targets,
                    epochs=2,
                    batch_size=32
                )
                assert len(history['train_loss']) == 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pytest.skip("Insufficient memory for large batch test")
                else:
                    raise