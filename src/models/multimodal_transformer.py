"""
multimodal_transformer.py
Unified Multi-Modal Cross-Attention Transformer and State-Space Dynamical Network.
Aligns continuous fMRI with high-density spike trains & LFP to decode latent cognitive states,
kinematics, and firing rates, while adapting to distribution shifts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatial_gnn import CoordinateAwareSpatialEncoder
from src.models.state_space_manifold import StateSpaceDynamicalModel
from src.models.cross_attention_transformer import MultiModalTransformerFusion
from src.models.contrastive_alignment import MultiModalContrastiveLoss, MaskedNeuralModelingLoss
from src.models.decoders import KinematicDecoder, PSTHReconstructionHead, DomainClassifier

class MultiModalNeuralFusionModel(nn.Module):
    """
    Complete Architecture:
    - Spatiotemporal GNN Encoders (3D Coordinate-Aware) + Direct Channel Residuals
    - State-Space Latent Manifold Generator (LFADS / continuous SSM)
    - Cross-Modal Transformer Fusion with Relative Temporal Bias
    - CPC Contrastive & Masked Neural Modeling Objectives
    - Kinematic & PSTH Decoders + Domain Adversarial Head
    """
    def __init__(
        self,
        n_units=48,
        n_rois=116,
        d_model=128,
        latent_dim=32,
        num_heads=4,
        num_layers=2,
        segment_sec=8.0,
        dt_spike=0.02,
        mask_ratio=0.15
    ):
        super().__init__()
        self.n_units = n_units
        self.n_rois = n_rois
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.segment_sec = segment_sec
        self.dt_spike = dt_spike

        # Direct channel residual projections (preserving unit-specific directional tuning)
        self.direct_spike_proj = nn.Sequential(
            nn.Linear(n_units * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.direct_fmri_proj = nn.Sequential(
            nn.Linear(n_rois, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 1. Spatial Geometry-Aware Encoders
        self.spike_spatial_encoder = CoordinateAwareSpatialEncoder(
            in_channels_per_node=2,
            hidden_dim=32,
            num_spatial_queries=4,
            out_dim=d_model,
            sigma=1.0 # Normalized coordinate scale
        )
        self.fmri_spatial_encoder = CoordinateAwareSpatialEncoder(
            in_channels_per_node=1,
            hidden_dim=32,
            num_spatial_queries=4,
            out_dim=d_model,
            sigma=1.0 # Normalized coordinate scale
        )

        # 2. Latent State-Space Manifold Model (LFADS / continuous SSM)
        self.ssm_manifold = StateSpaceDynamicalModel(
            input_dim=d_model,
            latent_dim=latent_dim,
            controller_dim=32,
            dt=dt_spike
        )

        # 3. Cross-Modal Fusion Transformer
        self.transformer_fusion = MultiModalTransformerFusion(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=256,
            dropout=0.1
        )

        # 4. Self-Supervised Objectives
        self.contrastive_head = MultiModalContrastiveLoss(
            spike_dim=d_model,
            fmri_dim=d_model,
            proj_dim=64,
            temperature=0.07
        )
        self.masked_modeling = MaskedNeuralModelingLoss(mask_ratio=mask_ratio)

        # 5. Output Decoders
        self.kinematic_decoder = KinematicDecoder(in_features=d_model, out_features=2, hidden_dim=64)
        self.psth_decoder = PSTHReconstructionHead(in_features=d_model, n_units=n_units, hidden_dim=64)
        self.domain_classifier = DomainClassifier(in_features=d_model, num_domains=2, hidden_dim=64)

    def forward(
        self,
        spikes,
        lfp,
        fmri,
        electrode_coords,
        roi_coords,
        apply_mask=False,
        return_interpretability=False
    ):
        """
        spikes: (B, T_s, n_units)
        lfp: (B, T_s, n_units)
        fmri: (B, T_f, n_rois)
        electrode_coords: (n_units, 3)
        roi_coords: (n_rois, 3)
        """
        B, T_s, _ = spikes.shape
        _, T_f, _ = fmri.shape

        # Optional Masked Neural Modeling
        if apply_mask and self.training:
            masked_spikes, mask = self.masked_modeling.apply_mask(spikes)
        else:
            masked_spikes, mask = spikes, None

        # 1. Direct channel projections + Spatiotemporal GNN Encodings
        spike_in_direct = torch.cat([masked_spikes, lfp], dim=-1)
        h_spike_direct = self.direct_spike_proj(spike_in_direct) # (B, T_s, d_model)

        spike_in_nodes = torch.stack([masked_spikes, lfp], dim=-1)
        h_spike_gnn, node_spikes = self.spike_spatial_encoder(spike_in_nodes, electrode_coords) # (B, T_s, d_model)
        h_spike_spatial = h_spike_direct + 0.5 * h_spike_gnn

        # fMRI stream
        h_fmri_direct = self.direct_fmri_proj(fmri) # (B, T_f, d_model)
        fmri_in_nodes = fmri.unsqueeze(-1)
        h_fmri_gnn, node_fmri = self.fmri_spatial_encoder(fmri_in_nodes, roi_coords) # (B, T_f, d_model)
        h_fmri_spatial = h_fmri_direct + 0.5 * h_fmri_gnn

        # 2. State-Space Latent Manifold Dynamics (LFADS / continuous SSM)
        ssm_factors, latent_traj, ssm_info = self.ssm_manifold(h_spike_spatial, return_dynamics=True)
        h_spike_ssm = h_spike_spatial + ssm_factors

        # 3. Cross-Attention Multimodal Transformer Fusion
        if return_interpretability:
            h_fused, cross_attn_weights = self.transformer_fusion(
                h_spike_ssm, h_fmri_spatial, segment_sec=self.segment_sec, return_attn=True
            )
        else:
            h_fused = self.transformer_fusion(
                h_spike_ssm, h_fmri_spatial, segment_sec=self.segment_sec, return_attn=False
            )
            cross_attn_weights = None

        # 4. Decoders
        kin_pred = self.kinematic_decoder(h_fused) # (B, T_s, 2)
        psth_pred = self.psth_decoder(h_fused)    # (B, T_s, n_units)
        domain_logits = self.domain_classifier(h_fused) # (B, 2)

        # Contrastive InfoNCE Loss
        contrastive_loss, contrastive_acc = self.contrastive_head(h_spike_ssm, h_fmri_spatial)

        # Masked Modeling Loss
        if mask is not None:
            mnm_loss = self.masked_modeling(psth_pred, spikes, mask)
        else:
            mnm_loss = torch.tensor(0.0, device=spikes.device)

        out = {
            "kin_pred": kin_pred,
            "psth_pred": psth_pred,
            "domain_logits": domain_logits,
            "latent_traj": latent_traj,
            "contrastive_loss": contrastive_loss,
            "contrastive_acc": contrastive_acc,
            "mnm_loss": mnm_loss,
            "kl_div": ssm_info["kl_div"],
            "dz_traj": ssm_info["dz_traj"],
            "h_fused": h_fused
        }

        if return_interpretability:
            out["cross_attn_weights"] = cross_attn_weights
            out["node_spikes"] = node_spikes
            out["node_fmri"] = node_fmri

        return out
