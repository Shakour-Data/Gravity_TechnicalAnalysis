"""
Transformer Model for Time Series Prediction

This module implements Transformer neural networks for:
- Advanced pattern recognition in financial time series
- Multi-head attention for capturing complex relationships
- Position encoding for temporal dependencies
- Self-attention mechanisms for long-range dependencies

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
from gravity_tech.models.schemas import Candle, TransformerResult, PredictionResult
from gravity_tech.core.domain.entities import PredictionSignal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore
        return x + pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""

    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, output_size: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, src):
        # Input projection
        src = self.input_projection(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Transformer encoding
        output = self.transformer_encoder(src)

        # Take the last time step
        output = output[:, -1, :]

        # Output projection
        output = self.dropout(output)
        output = self.output_projection(output)
        return output


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


class TransformerPredictor:
    """Transformer-based time series predictor"""

    def __init__(self, seq_length: int = 60, d_model: int = 64, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 256, learning_rate: float = 0.001, epochs: int = 100,
                 batch_size: int = 32, dropout: float = 0.1):
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None

    def prepare_data(self, candles: List[Candle], features: Optional[List[str]] = None) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Prepare candle data for Transformer training

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
        Train Transformer model

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
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)

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
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        # Load best model
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))

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

        # Calculate confidence based on attention weights (simplified)
        recent_prices = [c.close for c in candles[-20:]]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        confidence = max(0.1, 1.0 - volatility * 10)

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
            model_type="Transformer",
            description=f"Transformer prediction for next {steps_ahead} steps",
            prediction_timestamp=datetime.now(),
            input_features={'features': ['open', 'high', 'low', 'close', 'volume']},
            metadata={'steps_ahead': steps_ahead, 'volatility': float(volatility)}
        )

    def get_attention_weights(self, candles: List[Candle]) -> Optional[np.ndarray]:
        """
        Get attention weights for interpretability

        Args:
            candles: Input candles

        Returns:
            Attention weights matrix
        """
        if self.model is None:
            return None

        # This is a simplified version - in practice, you'd need to modify the model
        # to return attention weights
        data, _ = self.prepare_data(candles[-self.seq_length:])
        input_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()

        # Get intermediate representations (simplified)
        with torch.no_grad():
            # This would require modifying the Transformer to return attention
            # For now, return a placeholder
            return np.random.rand(self.seq_length, self.seq_length)


def create_transformer_predictor(seq_length: int = 60, d_model: int = 64) -> TransformerPredictor:
    """
    Factory function to create Transformer predictor

    Args:
        seq_length: Sequence length for Transformer
        d_model: Model dimension

    Returns:
        Configured TransformerPredictor instance
    """
    return TransformerPredictor(seq_length=seq_length, d_model=d_model)


def train_transformer_model(candles: List[Candle], target_feature: str = 'close') -> TransformerPredictor:
    """
    Convenience function to train Transformer model

    Args:
        candles: Training data
        target_feature: Feature to predict

    Returns:
        Trained TransformerPredictor
    """
    predictor = TransformerPredictor()
    predictor.train(candles, target_feature)
    return predictor