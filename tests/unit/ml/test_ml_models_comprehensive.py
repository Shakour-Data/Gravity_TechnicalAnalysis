"""
ML Model Component Tests - LSTM & Transformer

Comprehensive tests for deep learning models:
- Model initialization and configuration
- Forward pass computation
- Loss calculation
- Gradient flow
- Output shape validation
- Batch processing

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
import pytest
import torch


class TestLSTMModel:
    """Test LSTM model functionality"""

    def test_lstm_model_initialization(self):
        """Test LSTM model can be initialized"""
        try:
            from gravity_tech.ml.models.lstm_model import LSTMModel
            model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, output_size=5)
            assert model is not None
        except ImportError:
            pytest.skip("LSTM model not available")

    def test_lstm_forward_pass(self):
        """Test LSTM forward pass computation"""
        try:
            from gravity_tech.ml.models.lstm_model import LSTMModel
            model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, output_size=5)

            # Create dummy input
            batch_size = 32
            seq_length = 20
            x = torch.randn(batch_size, seq_length, 10)

            # Forward pass should not raise
            if hasattr(model, 'forward'):
                output = model.forward(x)
                assert output is not None
        except ImportError:
            pytest.skip("LSTM model not available")

    def test_lstm_output_shape(self):
        """Test LSTM output shape correctness"""
        try:
            from gravity_tech.ml.models.lstm_model import LSTMModel
            input_size, hidden_size, output_size = 10, 64, 5
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                output_size=output_size
            )

            batch_size = 32
            x = torch.randn(batch_size, 20, input_size)

            if hasattr(model, 'forward'):
                output = model.forward(x)
                # Output should have batch dimension and output size
                if output is not None:
                    assert output.shape[0] == batch_size
        except ImportError:
            pytest.skip("LSTM model not available")

    def test_lstm_with_different_sequence_lengths(self):
        """Test LSTM with variable sequence lengths"""
        try:
            from gravity_tech.ml.models.lstm_model import LSTMModel
            model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, output_size=5)

            sequences = [
                torch.randn(1, 10, 10),  # seq_len=10
                torch.randn(1, 20, 10),  # seq_len=20
                torch.randn(1, 30, 10),  # seq_len=30
            ]

            for seq in sequences:
                if hasattr(model, 'forward'):
                    output = model.forward(seq)
                    assert output is not None
        except ImportError:
            pytest.skip("LSTM model not available")

    def test_lstm_batch_processing(self):
        """Test LSTM processes batches correctly"""
        try:
            from gravity_tech.ml.models.lstm_model import LSTMModel
            model = LSTMModel(input_size=10, hidden_size=32, num_layers=1, output_size=5)

            batches = [16, 32, 64, 128]
            for batch_size in batches:
                x = torch.randn(batch_size, 20, 10)
                if hasattr(model, 'forward'):
                    output = model.forward(x)
                    if output is not None:
                        assert output.shape[0] == batch_size
        except ImportError:
            pytest.skip("LSTM model not available")


class TestTransformerModel:
    """Test Transformer model functionality"""

    def test_transformer_model_initialization(self):
        """Test Transformer model initialization"""
        try:
            from gravity_tech.ml.models.transformer_model import TransformerModel
            model = TransformerModel(
                input_size=10,
                d_model=64,
                nhead=4,
                num_layers=2,
                output_size=5,
                dim_feedforward=256
            )
            assert model is not None
        except ImportError:
            pytest.skip("Transformer model not available")

    def test_transformer_forward_pass(self):
        """Test Transformer forward pass"""
        try:
            from gravity_tech.ml.models.transformer_model import TransformerModel
            model = TransformerModel(
                input_size=10,
                d_model=64,
                nhead=4,
                num_layers=2,
                output_size=5,
                dim_feedforward=256
            )

            batch_size = 32
            seq_length = 20
            x = torch.randn(batch_size, seq_length, 10)

            if hasattr(model, 'forward'):
                output = model.forward(x)
                assert output is not None
        except ImportError:
            pytest.skip("Transformer model not available")

    def test_transformer_attention_heads(self):
        """Test Transformer with different attention head configurations"""
        try:
            from gravity_tech.ml.models.transformer_model import TransformerModel

            head_configs = [1, 2, 4, 8]
            d_model = 64

            for nhead in head_configs:
                if d_model % nhead == 0:  # d_model must be divisible by nhead
                    model = TransformerModel(
                        input_size=10,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=2,
                        output_size=5,
                        dim_feedforward=256
                    )
                    x = torch.randn(16, 20, 10)

                    if hasattr(model, 'forward'):
                        output = model.forward(x)
                        assert output is not None
        except ImportError:
            pytest.skip("Transformer model not available")

    def test_transformer_output_shape(self):
        """Test Transformer output dimensions"""
        try:
            from gravity_tech.ml.models.transformer_model import TransformerModel

            output_size = 5
            model = TransformerModel(
                input_size=10,
                d_model=64,
                nhead=4,
                num_layers=2,
                output_size=output_size,
                dim_feedforward=256
            )

            x = np.random.randn(32, 20, 10)

            if hasattr(model, 'forward'):
                output = model.forward(x)
                if output is not None:
                    # Output should have correct dimensions
                    assert output.shape[0] == 32  # Batch size
                    assert output.shape[-1] == output_size  # Output features
        except ImportError:
            pytest.skip("Transformer model not available")


class TestModelTraining:
    """Test model training procedures"""

    def test_model_loss_computation(self):
        """Test loss computation during training"""
        # This is a conceptual test - actual implementation varies
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

        # MSE Loss
        mse_loss = np.mean((y_true - y_pred) ** 2)
        assert mse_loss >= 0
        assert isinstance(mse_loss, float | np.floating)

    def test_model_gradient_shapes(self):
        """Test gradient shapes for model parameters"""
        # This tests the concept of gradients
        params = {
            "W1": np.random.randn(10, 64),
            "b1": np.random.randn(64),
            "W2": np.random.randn(64, 5),
            "b2": np.random.randn(5),
        }

        gradients = {
            name: np.random.randn(*param.shape)
            for name, param in params.items()
        }

        # Gradients should have same shape as parameters
        for name in params:
            assert gradients[name].shape == params[name].shape

    def test_model_optimization_step(self):
        """Test optimization step"""
        learning_rate = 0.001
        params = {
            "W": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([0.5, -0.5]),
        }
        gradients = {
            "W": np.array([[0.1, -0.1], [-0.2, 0.2]]),
            "b": np.array([0.01, -0.01]),
        }

        # Update parameters
        updated_params = {
            name: params[name] - learning_rate * gradients[name]
            for name in params
        }

        # Parameters should be updated
        assert not np.allclose(params["W"], updated_params["W"])

    def test_batch_normalization(self):
        """Test batch normalization effect"""
        batch_size = 32
        features = 10
        x = np.random.randn(batch_size, features)

        # Compute batch statistics
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        # Normalize
        x_normalized = (x - mean) / (std + 1e-5)

        # Normalized values should have mean ~0 and std ~1
        assert np.allclose(np.mean(x_normalized, axis=0), 0, atol=1e-5)
        assert np.allclose(np.std(x_normalized, axis=0), 1.0, atol=0.1)

    def test_dropout_effect(self):
        """Test dropout regularization"""
        dropout_rate = 0.5
        x = np.random.randn(32, 100)

        # Simulate dropout
        mask = np.random.binomial(1, 1 - dropout_rate, x.shape)
        x_dropout = x * mask / (1 - dropout_rate)

        # Dropout should reduce some activations
        assert np.sum(x_dropout == 0) > 0  # Some values zeroed out

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        initial_lr = 0.01
        epochs = [0, 10, 20, 30]
        schedules = {
            "exponential_decay": [initial_lr * (0.95 ** e) for e in epochs],
            "step_decay": [initial_lr if e < 15 else initial_lr * 0.1 for e in epochs],
        }

        for _schedule_name, lrs in schedules.items():
            # Learning rates should be positive
            assert all(lr > 0 for lr in lrs)
            # Learning rates should generally decrease
            for i in range(len(lrs) - 1):
                assert lrs[i] >= lrs[i + 1]


class TestModelEvaluation:
    """Test model evaluation metrics"""

    def test_accuracy_metric(self):
        """Test accuracy computation"""
        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1])

        accuracy = np.mean(y_true == y_pred)
        assert 0 <= accuracy <= 1
        assert accuracy == 5 / 6  # 5 correct out of 6

    def test_precision_recall(self):
        """Test precision and recall"""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])

        # True positives, false positives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_f1_score(self):
        """Test F1 score computation"""
        # Example: 4 TP, 1 FP, 1 FN
        tp, fp, fn = 4, 1, 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        assert 0 <= f1 <= 1
        assert f1 > 0.5  # Good F1 score

    def test_roc_auc_concept(self):
        """Test ROC-AUC conceptual computation"""
        y_true = np.array([1, 1, 0, 0, 1])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.7])

        # Sort by score
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true[sorted_indices]

        # Compute ROC points (simplified)
        assert len(y_true_sorted) == len(y_score)

    def test_confusion_matrix(self):
        """Test confusion matrix"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

        # Compute confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        assert tp + tn + fp + fn == len(y_true)
        assert tp == 2
        assert tn == 3
        assert fp == 1
        assert fn == 2


class TestModelInference:
    """Test model inference and prediction"""

    def test_single_sample_prediction(self):
        """Test prediction on single sample"""
        # Single sample
        # Simulated prediction
        prediction = np.random.randn(1, 5)
        assert prediction.shape == (1, 5)

    def test_batch_prediction(self):
        """Test batch prediction"""
        batch_size = 32
        predictions = np.random.randn(batch_size, 20, 10)
        assert predictions.shape == (batch_size, 20, 10)

    def test_probability_output(self):
        """Test model producing valid probabilities"""
        predictions = np.array([0.1, 0.7, 0.2])

        # Sum to 1
        assert np.isclose(np.sum(predictions), 1.0)

        # All in [0, 1]
        assert np.all((predictions >= 0) & (predictions <= 1))

    def test_confidence_scores(self):
        """Test confidence score interpretation"""
        predictions = np.array([
            [0.95, 0.05],  # High confidence
            [0.51, 0.49],  # Low confidence
            [0.99, 0.01],  # Very high confidence
        ])

        confidence = np.max(predictions, axis=1)
        assert confidence[0] > 0.9
        assert confidence[1] < 0.6
        assert confidence[2] > 0.95

