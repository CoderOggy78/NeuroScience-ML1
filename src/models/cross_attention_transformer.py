"""
cross_attention_transformer.py
Multi-Modal Cross-Attention Fusion Transformer with Continuous Temporal Relative Bias.
Aligns low-temporal/high-spatial fMRI BOLD streams with high-temporal/sparse-spatial
spike trains and LFP to reconstruct latent cognitive states and kinematics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousTemporalRelativeBias(nn.Module):
    """
    Biophysically-inspired Continuous Temporal Relative Positional Bias.
    Computes time-lag delta_t = t_spike - t_fmri and projects through a Gaussian RBF bank
    to model the canonical Hemodynamic Response Function (HRF) delay (peaking at ~5s).
    """
    def __init__(self, num_heads=4, num_rbf=16, max_lag_sec=16.0):
        super().__init__()
        self.num_heads = num_heads
        self.centers = nn.Parameter(torch.linspace(-max_lag_sec, max_lag_sec, num_rbf), requires_grad=False)
        self.log_gamma = nn.Parameter(torch.ones(num_rbf) * 0.5)
        self.rbf_proj = nn.Linear(num_rbf, num_heads)

    def forward(self, t_spike, t_fmri):
        """
        t_spike: (T_spike,) continuous timestamps in seconds
        t_fmri: (T_fmri,) continuous timestamps in seconds
        Returns:
            bias: (1, num_heads, T_spike, T_fmri)
        """
        delta_t = t_spike.unsqueeze(1) - t_fmri.unsqueeze(0) # (T_s, T_f)
        diff = delta_t.unsqueeze(-1) - self.centers.view(1, 1, -1) # (T_s, T_f, RBF)
        gamma = torch.exp(self.log_gamma).view(1, 1, -1)
        rbf_features = torch.exp(-gamma * (diff ** 2)) # (T_s, T_f, RBF)
        bias = self.rbf_proj(rbf_features) # (T_s, T_f, num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0) # (1, num_heads, T_s, T_f)
        return bias


class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal multi-head attention block where high-temporal spike queries
    attend over low-temporal fMRI key/values with continuous relative temporal bias.
    """
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.temporal_bias = ContinuousTemporalRelativeBias(num_heads=num_heads)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_spike, kv_fmri, t_spike, t_fmri, return_attn=False):
        """
        q_spike: (B, T_spike, d_model)
        kv_fmri: (B, T_fmri, d_model)
        t_spike: (T_spike,) timestamps in sec
        t_fmri: (T_fmri,) timestamps in sec
        """
        B, T_s, D = q_spike.shape
        _, T_f, _ = kv_fmri.shape

        q_norm = self.norm_q(q_spike)
        kv_norm = self.norm_kv(kv_fmri)

        Q = self.q_proj(q_norm).view(B, T_s, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T_s, d_k)
        K = self.k_proj(kv_norm).view(B, T_f, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T_f, d_k)
        V = self.v_proj(kv_norm).view(B, T_f, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T_f, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T_s, T_f)
        
        # Add continuous biophysical temporal bias (HRF time-lag prior)
        bias = self.temporal_bias(t_spike, t_fmri) # (1, H, T_s, T_f)
        scores = scores + bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_drop, V) # (B, H, T_s, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T_s, D)
        out = self.out_proj(context)

        if return_attn:
            return out, attn_weights
        return out


class MultiModalTransformerFusion(nn.Module):
    """
    Complete Multi-Modal Cross-Attention Transformer and State-Space Hybrid.
    Integrates:
    - Cross-modal attention aligning spike queries with fMRI keys/values
    - Continuous HRF time-lag relative bias
    - State-Space sequence dynamics (SSM) modeling non-linear neural manifolds
    """
    def __init__(self, d_model=128, num_heads=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.cross_attn = CrossModalAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        # Recurrent State-Space Temporal Dynamics Engine (SSM)
        self.state_space_temporal = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, h_spike, h_fmri, segment_sec=8.0, return_attn=False):
        """
        h_spike: (B, T_spike, d_model)
        h_fmri: (B, T_fmri, d_model)
        segment_sec: total duration in seconds
        """
        B, T_s, D = h_spike.shape
        _, T_f, _ = h_fmri.shape
        device = h_spike.device

        # Continuous timestamps
        t_spike = torch.linspace(0, segment_sec, T_s, device=device)
        t_fmri = torch.linspace(0, segment_sec, T_f, device=device)

        # Cross-modal alignment
        if return_attn:
            fused_context, attn_weights = self.cross_attn(h_spike, h_fmri, t_spike, t_fmri, return_attn=True)
        else:
            fused_context = self.cross_attn(h_spike, h_fmri, t_spike, t_fmri, return_attn=False)
            attn_weights = None

        # Residual connection
        h_fused = self.norm(h_spike + fused_context)

        # Deep temporal state-space modeling across the fused sequence
        ssm_out, _ = self.state_space_temporal(h_fused)
        out = self.norm(ssm_out + self.mlp(ssm_out))

        if return_attn:
            return out, attn_weights
        return out
