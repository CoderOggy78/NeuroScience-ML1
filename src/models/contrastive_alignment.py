"""
contrastive_alignment.py
Contrastive Predictive Coding (CPC / InfoNCE) and Masked Neural Modeling (MNM) Objectives
for self-supervised multi-modal neural representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalContrastiveLoss(nn.Module):
    """
    InfoNCE / Contrastive Predictive Coding Loss aligning Spike and fMRI latent dynamics.
    Aligns temporally co-occurring segments across modalities in a shared continuous embedding space.
    """
    def __init__(self, spike_dim=128, fmri_dim=128, proj_dim=64, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.spike_proj = nn.Sequential(
            nn.Linear(spike_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(fmri_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, h_spike, h_fmri):
        """
        h_spike: (B, T_spike, spike_dim)
        h_fmri: (B, T_fmri, fmri_dim)
        Temporal mean pooling or window sampling for contrastive alignment
        """
        # Pool across time to get window-level representations
        z_s = torch.mean(h_spike, dim=1) # (B, spike_dim)
        z_f = torch.mean(h_fmri, dim=1)  # (B, fmri_dim)

        p_s = F.normalize(self.spike_proj(z_s), dim=-1) # (B, proj_dim)
        p_f = F.normalize(self.fmri_proj(z_f), dim=-1) # (B, proj_dim)

        # Cosine similarity matrix
        sim = torch.matmul(p_s, p_f.T) / self.temperature # (B, B)
        labels = torch.arange(p_s.size(0), device=p_s.device)

        loss_s2f = F.cross_entropy(sim, labels)
        loss_f2s = F.cross_entropy(sim.T, labels)
        contrastive_loss = (loss_s2f + loss_f2s) / 2.0

        # Contrastive accuracy
        with torch.no_grad():
            preds = torch.argmax(sim, dim=-1)
            accuracy = (preds == labels).float().mean().item()

        return contrastive_loss, accuracy


class MaskedNeuralModelingLoss(nn.Module):
    """
    Masked Neural Modeling (MNM):
    Stochastically masks a proportion of electrode channels / time tokens
    and trains the network to reconstruct the missing neural population activity.
    """
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def apply_mask(self, spikes):
        """
        spikes: (B, T, N_units)
        Returns:
            masked_spikes: (B, T, N_units)
            mask: (B, T, N_units) boolean tensor of masked positions
        """
        B, T, N = spikes.shape
        # Create Bernoulli random mask
        mask = torch.rand(B, T, N, device=spikes.device) < self.mask_ratio
        masked_spikes = spikes.clone()
        masked_spikes[mask] = 0.0
        return masked_spikes, mask

    def forward(self, reconstructed_rates, target_spikes, mask):
        """
        Poisson Negative Log-Likelihood over masked units:
        loss = recon - target * log(recon + eps)
        """
        eps = 1e-6
        poisson_loss = reconstructed_rates - target_spikes * torch.log(reconstructed_rates + eps)
        # Compute loss exclusively on masked tokens
        if mask.sum() > 0:
            masked_loss = poisson_loss[mask].mean()
        else:
            masked_loss = poisson_loss.mean()
        return masked_loss
