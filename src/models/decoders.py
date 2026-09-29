"""
decoders.py
Decoding Heads:
1. Continuous Kinematic Decoder (Velocity/Speed reconstruction).
2. Firing Rate (PSTH) Intensity Prediction Head.
3. Domain Adaptation Head with Gradient Reversal Layer (GRL) for electrode drift invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class KinematicDecoder(nn.Module):
    """
    Decodes continuous kinematic movement (velocity vx, vy) from aligned neural representations.
    """
    def __init__(self, in_features=128, out_features=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, h):
        """
        h: (B, T, in_features)
        Returns:
            kin_pred: (B, T, out_features)
        """
        return self.net(h)


class PSTHReconstructionHead(nn.Module):
    """
    Reconstructs instantaneous single-unit firing rate intensities (PSTH) lambda_t
    using a log-linear / Poisson intensity model: lambda = exp(W * h + b)
    """
    def __init__(self, in_features=128, n_units=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_units)
        )

    def forward(self, h):
        """
        h: (B, T, in_features)
        Returns:
            rates: (B, T, n_units) strictly positive firing rates
        """
        log_rates = self.net(h)
        rates = torch.exp(torch.clamp(log_rates, -10.0, 5.0))
        return rates


class DomainClassifier(nn.Module):
    """
    Discriminates recording session / drift domain (Session 0 vs Drift Session 1).
    Used in tandem with Gradient Reversal to force latent representations to be invariant
    to session shift, electrode drift, and fatigue.
    """
    def __init__(self, in_features=128, num_domains=2, hidden_dim=64, alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_domains)
        )

    def set_alpha(self, alpha):
        self.grl.alpha = alpha

    def forward(self, h):
        """
        h: (B, T, in_features)
        Pool across time and pass through GRL
        """
        h_pool = torch.mean(h, dim=1) # (B, in_features)
        h_rev = self.grl(h_pool)
        domain_logits = self.net(h_rev)
        return domain_logits
