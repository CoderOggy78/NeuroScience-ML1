"""
Unit tests for evaluation.py:
- Metrics calculation (R^2, RMSE, Pearson r, TDS)
- Baseline neural decoders (Linear, LSTM, Transformer)
"""

import pytest
import numpy as np
import torch

from src.evaluation import (
    compute_kinematic_r2,
    compute_kinematic_rmse,
    compute_psth_correlation,
    compute_transfer_degradation_score,
    LinearDecoderBaseline,
    SingleModalityLSTM,
    SingleModalityTransformer
)


def test_kinematic_r2_and_rmse():
    """Verify R^2 and RMSE on synthetic perfect and noisy predictions."""
    y_true = np.sin(np.linspace(0, 10, 200))[:, None]
    y_true = np.tile(y_true, (1, 2))  # (200, 2)
    
    # Perfect prediction
    r2_perf, r2_dims = compute_kinematic_r2(y_true, y_true)
    rmse_perf = compute_kinematic_rmse(y_true, y_true)
    assert np.isclose(r2_perf, 1.0)
    assert np.isclose(rmse_perf, 0.0)

    # Constant prediction should yield R^2 ~ 0.0
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    y_mean_pred = np.tile(y_mean, (200, 1))
    r2_mean, _ = compute_kinematic_r2(y_true, y_mean_pred)
    assert np.isclose(r2_mean, 0.0, atol=1e-3)


def test_transfer_degradation_score():
    """Verify TDS calculation across different degradation scenarios."""
    # Zero degradation: source 0.8, target 0.8 -> 0%
    assert compute_transfer_degradation_score(0.8, 0.8) == 0.0

    # 50% degradation: source 0.8, target 0.4 -> 50%
    assert np.isclose(compute_transfer_degradation_score(0.8, 0.4), 50.0)

    # Catastrophic failure: source 0.8, target -0.2 -> 125%
    assert np.isclose(compute_transfer_degradation_score(0.8, -0.2), 125.0)

    # Negative source baseline
    assert compute_transfer_degradation_score(-0.1, -0.2) == 100.0


def test_baseline_models_forward():
    """Verify forward pass shapes for Linear, LSTM, and Transformer baselines."""
    B, T, in_features = 2, 50, 32
    spikes = torch.randn(B, T, in_features)

    linear_dec = LinearDecoderBaseline(in_features=in_features, out_features=2)
    y_lin = linear_dec(spikes)
    assert y_lin.shape == (B, T, 2)

    lstm_dec = SingleModalityLSTM(in_features=in_features, hidden_dim=32, out_features=2, num_layers=1)
    y_lstm = lstm_dec(spikes)
    assert y_lstm.shape == (B, T, 2)

    tf_dec = SingleModalityTransformer(in_features=in_features, d_model=32, nhead=2, num_layers=1, out_features=2)
    y_tf = tf_dec(spikes)
    assert y_tf.shape == (B, T, 2)
