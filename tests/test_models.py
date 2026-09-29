"""
Unit tests for neural architectures:
- Spatial GNN with 3D coordinate-aware attention
- State-space neural manifold generator (LFADS / continuous SSM)
- Cross-modal attention transformer with continuous relative temporal bias
- Output decoders (kinematic, Poisson PSTH, domain classifier with GRL)
- Complete unified MultiModalNeuralFusionModel
"""

import pytest
import torch
import torch.nn as nn

from src.models.spatial_gnn import CoordinateAwareSpatialEncoder
from src.models.state_space_manifold import StateSpaceDynamicalModel
from src.models.cross_attention_transformer import MultiModalTransformerFusion
from src.models.decoders import KinematicDecoder, PSTHReconstructionHead, DomainClassifier, GradientReversalLayer
from src.models.multimodal_transformer import MultiModalNeuralFusionModel


def test_spatial_gnn_encoder():
    """Verify coordinate-aware spatial GNN forward pass and output dimensionality."""
    B, T, N, C = 2, 50, 16, 2
    encoder = CoordinateAwareSpatialEncoder(
        in_channels_per_node=C,
        hidden_dim=32,
        num_spatial_queries=4,
        out_dim=64,
        sigma=1.0
    )
    x = torch.randn(B, T, N, C)
    coords = torch.randn(N, 3)

    out, node_feats = encoder(x, coords)
    assert out.shape == (B, T, 64)
    assert node_feats.shape == (B, T, N, 32)


def test_state_space_manifold_dynamics():
    """Verify continuous state-space sequence generator and phase space outputs."""
    B, T, D_in = 2, 50, 64
    latent_dim = 16
    ssm = StateSpaceDynamicalModel(input_dim=D_in, latent_dim=latent_dim, controller_dim=32, dt=0.02)
    x = torch.randn(B, T, D_in)

    factors, latent_traj, info = ssm(x, return_dynamics=True)
    assert factors.shape == (B, T, D_in)
    assert latent_traj.shape == (B, T, latent_dim)
    assert info["dz_traj"].shape == (B, T, latent_dim)
    assert "kl_div" in info


def test_cross_modal_transformer_with_hrf_prior():
    """Verify cross-attention with relative temporal bias and attention weights extraction."""
    B, T_s, T_f, d_model = 2, 50, 4, 64
    x_spike = torch.randn(B, T_s, d_model)
    x_fmri = torch.randn(B, T_f, d_model)

    fusion = MultiModalTransformerFusion(d_model=d_model, num_heads=4, num_layers=2)

    # Test forward pass with attention extraction
    h_fused, attn_weights = fusion(x_spike, x_fmri, segment_sec=8.0, return_attn=True)
    assert h_fused.shape == (B, T_s, d_model)
    assert attn_weights is not None
    # Attn weights shape: (B, num_heads, T_spike, T_fmri)
    assert attn_weights.shape == (B, 4, T_s, T_f)


def test_gradient_reversal_layer():
    """Verify GradientReversalLayer flips backward gradient sign."""
    alpha = 0.75
    grl = GradientReversalLayer(alpha=alpha)
    x = torch.tensor([[2.0, 3.0]], requires_grad=True)
    y = grl(x)
    # Forward is identity
    assert torch.allclose(x, y)

    loss = (y * 2.0).sum()
    loss.backward()
    # Backward gradient should be -2.0 * alpha = -1.5
    assert torch.allclose(x.grad, torch.tensor([[-1.5, -1.5]]))


def test_full_multimodal_model_forward():
    """Verify unified MultiModalNeuralFusionModel end-to-end forward pass."""
    B, T_s, T_f = 2, 40, 4
    n_units = 16
    n_rois = 30
    d_model = 64

    model = MultiModalNeuralFusionModel(
        n_units=n_units,
        n_rois=n_rois,
        d_model=d_model,
        latent_dim=16,
        num_heads=2,
        num_layers=1,
        segment_sec=8.0,
        dt_spike=0.02
    )

    spikes = torch.rand(B, T_s, n_units)
    lfp = torch.randn(B, T_s, n_units)
    fmri = torch.randn(B, T_f, n_rois)
    electrode_coords = torch.randn(n_units, 3)
    roi_coords = torch.randn(n_rois, 3)

    model.eval()
    with torch.no_grad():
        out = model(
            spikes,
            lfp,
            fmri,
            electrode_coords,
            roi_coords,
            return_interpretability=True
        )

    assert out["kin_pred"].shape == (B, T_s, 2)
    assert out["psth_pred"].shape == (B, T_s, n_units)
    assert out["domain_logits"].shape == (B, 2)
    assert out["latent_traj"].shape == (B, T_s, 16)
    assert out["cross_attn_weights"].shape == (B, 2, T_s, T_f)
    assert out["contrastive_loss"].item() >= 0.0
