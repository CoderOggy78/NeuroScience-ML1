"""
evaluation.py
Comprehensive Evaluation, Baselines, and Metrics Suite:
- Kinematic R^2 and RMSE
- Single-unit PSTH Firing Rate Correlation (Pearson r)
- Transfer Degradation Score (TDS) under electrode drift and session non-stationarity
- Baseline Architectures:
    1. Linear Ridge / Kalman Filter Baseline
    2. Single-Modality LSTM (Spikes only)
    3. Single-Modality Transformer (Spikes only)
    4. Proposed Multi-Modal Cross-Attention Transformer + SSM
    5. Proposed + GRL Domain Adaptation (Ours + DA)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# ----------------- Baseline Architectures ----------------- #

class LinearDecoderBaseline(nn.Module):
    """
    Standard Linear / Ridge regression decoder mapping binned spike counts
    directly to continuous kinematics.
    """
    def __init__(self, in_features=48, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, spikes):
        """
        spikes: (B, T, in_features)
        Returns: (B, T, out_features)
        """
        return self.linear(spikes)


class SingleModalityLSTM(nn.Module):
    """
    Single-Modality Recurrent Decoder (LSTM) on high-density spike trains.
    """
    def __init__(self, in_features=48, hidden_dim=64, out_features=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, out_features)

    def forward(self, spikes):
        out, _ = self.lstm(spikes)
        return self.head(out)


class SingleModalityTransformer(nn.Module):
    """
    Single-Modality Transformer Decoder on high-density spike trains.
    """
    def __init__(self, in_features=48, d_model=64, nhead=4, num_layers=2, out_features=2):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_features)

    def forward(self, spikes):
        h = self.in_proj(spikes)
        out = self.transformer(h)
        return self.head(out)

# ----------------- Metrics Utilities ----------------- #

def compute_kinematic_r2(y_true, y_pred):
    """
    Computes R^2 (coefficient of determination) averaged across kinematic dimensions (vx, vy).
    y_true: (N, T, D) or (N * T, D)
    y_pred: (N, T, D) or (N * T, D)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    
    r2_dims = []
    for d in range(y_true_flat.shape[-1]):
        val = r2_score(y_true_flat[:, d], y_pred_flat[:, d])
        r2_dims.append(val)
    return float(np.mean(r2_dims)), [float(v) for v in r2_dims]


def compute_kinematic_rmse(y_true, y_pred):
    """
    Computes Root Mean Square Error (RMSE) across kinematic dimensions.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    return float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))


def compute_psth_correlation(true_rates, pred_rates):
    """
    Computes mean Pearson correlation r between ground-truth and reconstructed firing rates across units.
    """
    if isinstance(true_rates, torch.Tensor):
        true_rates = true_rates.detach().cpu().numpy()
    if isinstance(pred_rates, torch.Tensor):
        pred_rates = pred_rates.detach().cpu().numpy()

    true_flat = true_rates.reshape(-1, true_rates.shape[-1])
    pred_flat = pred_rates.reshape(-1, pred_rates.shape[-1])
    
    corrs = []
    for u in range(true_flat.shape[-1]):
        if np.std(true_flat[:, u]) > 1e-6 and np.std(pred_flat[:, u]) > 1e-6:
            r, _ = pearsonr(true_flat[:, u], pred_flat[:, u])
            if not np.isnan(r):
                corrs.append(r)
    return float(np.mean(corrs)) if len(corrs) > 0 else 0.0


def compute_transfer_degradation_score(r2_source, r2_target):
    """
    Transfer Degradation Score (TDS): Percentage drop in decoding performance under session shift:
    TDS = max(0, (R2_source - R2_target) / R2_source) * 100%
    Lower TDS is better (0% means zero degradation).
    """
    if r2_source <= 0:
        return 100.0
    deg = (r2_source - r2_target) / r2_source * 100.0
    return float(max(0.0, deg))
