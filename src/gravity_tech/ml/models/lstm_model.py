"""
LSTM Model for Time Series Prediction

This module implements LSTM (Long Short-Term Memory) neural networks for:
- Price prediction and forecasting
- Volatility prediction
- Trend continuation/probability analysis
- Multi-step ahead predictions

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Any
from gravity_tech.models.schemas import Candle, LSTMResult, PredictionResult
from gravity_tech.core.domain.entities import PredictionSignal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""

    def __init__(self, data: np.ndarray, seq_length: int, target_idx: int = -1):
        self.data = data
        self.seq_length = seq_length
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length, self.target_idx] if self.target_idx != -1 else self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    """LSTM Neural Network for time series prediction"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last time step
        out = self.fc(out)
        return out


class LSTMPredictor:
    """LSTM-based time series predictor"""

    def __init__(self, seq_length: int = 60, hidden_size: int = 64, num_layers: int = 2,
                 learning_rate: float = 0.001, epochs: int = 100, batch_size: int = 32):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None

    def prepare_data(self, candles: List[Candle], features: Optional[List[str]] = None) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Prepare candle data for LSTM training

        Args:
            candles: List of candles
            features: List of features to use (default: OHLCV)

        Returns:
            Prepared data array and scaler
        """
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']

        data = []
        for candle in candles:
            row = []
            if 'open' in features:
                row.append(candle.open)
            if 'high' in features:
                row.append(candle.high)
            if 'low' in features:
                row.append(candle.low)
            if 'close' in features:
                row.append(candle.close)
            if 'volume' in features:
                row.append(candle.volume)
            data.append(row)

        data = np.array(data)
        self.input_size = len(features)

        # Scale data
        scaled_data = self.scaler.fit_transform(data)

        return scaled_data, self.scaler

    def train(self, candles: List[Candle], target_feature: str = 'close',
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train LSTM model

        Args:
            candles: Training candles
            target_feature: Feature to predict
            validation_split: Validation data ratio

        Returns:
            Training metrics
        """
        # Prepare data
        data, _ = self.prepare_data(candles)

        # Create target data (next close price)
        target_idx = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4}.get(target_feature, 3)

        # Split data
        train_size = int(len(data) * (1 - validation_split))
        train_data = data[:train_size]
        val_data = data[train_size:]

        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.seq_length, target_idx)
        val_dataset = TimeSeriesDataset(val_data, self.seq_length, target_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        input_size = self.input_size if self.input_size is not None else 5
        self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, 1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            epoch_val_loss = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(x_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    epoch_val_loss += loss.item()

            epoch_val_loss /= len(val_loader)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_loss,
            'epochs_trained': len(train_losses)
        }

    def predict(self, candles: List[Candle], steps_ahead: int = 1) -> PredictionResult:
        """
        Make predictions using trained model

        Args:
            candles: Recent candles for prediction
            steps_ahead: Number of steps to predict ahead

        Returns:
            PredictionResult with predictions and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare input data
        data, _ = self.prepare_data(candles[-self.seq_length:])
        input_seq = torch.tensor(data[-self.seq_length:], dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(steps_ahead):
                pred = self.model(input_seq)
                pred_value = pred.item()

                predictions.append(pred_value)

                # Update input sequence for multi-step prediction
                if steps_ahead > 1:
                    # Create new input by appending prediction and removing oldest
                    input_size = self.input_size if self.input_size is not None else 5
                    new_input = np.append(input_seq.cpu().numpy()[0, 1:], [[pred_value] * input_size], axis=0)
                    input_seq = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inverse transform predictions
        pred_array = np.array(predictions).reshape(-1, 1)
        input_size = self.input_size if self.input_size is not None else 5
        predictions_unscaled = self.scaler.inverse_transform(
            np.tile(pred_array, (1, input_size))
        )[:, 0]  # Take first column (close price)

        # Calculate confidence based on recent volatility
        recent_prices = [c.close for c in candles[-20:]]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        confidence = max(0.1, 1.0 - volatility * 10)  # Lower confidence with higher volatility

        # Determine signal
        current_price = candles[-1].close
        predicted_price = predictions_unscaled[0]

        if predicted_price > current_price * 1.005:  # 0.5% threshold
            signal = PredictionSignal.BULLISH
        elif predicted_price < current_price * 0.995:
            signal = PredictionSignal.BEARISH
        else:
            signal = PredictionSignal.NEUTRAL

        from datetime import datetime
        return PredictionResult(
            predictions=list(predictions_unscaled),
            confidence=float(confidence),
            signal=signal,
            model_type="LSTM",
            description=f"LSTM prediction for next {steps_ahead} steps",
            prediction_timestamp=datetime.now(),
            input_features={'features': ['open', 'high', 'low', 'close', 'volume']},
            metadata={'steps_ahead': steps_ahead, 'volatility': float(volatility)}
        )

    def predict_volatility(self, candles: List[Candle], window: int = 20) -> float:
        """
        Predict future volatility using LSTM

        Args:
            candles: Historical candles
            window: Rolling window for volatility calculation

        Returns:
            Predicted volatility
        """
        if len(candles) < window + self.seq_length:
            return 0.0

        # Calculate historical volatility
        returns = []
        for i in range(window, len(candles)):
            ret = (candles[i].close - candles[i-window].close) / candles[i-window].close
            returns.append(ret)

        # Prepare data for LSTM (use returns as features)
        data = np.array(returns).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        # Make prediction
        input_seq = scaled_data[-self.seq_length:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_tensor)
            pred_scaled = pred.item()

        # Inverse transform
        pred_volatility_array = self.scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]

        return abs(pred_volatility_array)  # Return absolute value


def create_lstm_predictor(seq_length: int = 60, hidden_size: int = 64) -> LSTMPredictor:
    """
    Factory function to create LSTM predictor

    Args:
        seq_length: Sequence length for LSTM
        hidden_size: Hidden layer size

    Returns:
        Configured LSTMPredictor instance
    """
    return LSTMPredictor(seq_length=seq_length, hidden_size=hidden_size)


def train_lstm_model(candles: List[Candle], target_feature: str = 'close') -> LSTMPredictor:
    """
    Convenience function to train LSTM model

    Args:
        candles: Training data
        target_feature: Feature to predict

    Returns:
        Trained LSTMPredictor
    """
    predictor = LSTMPredictor()
    predictor.train(candles, target_feature)
    return predictor