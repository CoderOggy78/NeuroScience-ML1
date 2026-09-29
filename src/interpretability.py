"""
interpretability.py
Biophysical and Latent Dynamical Interpretability Suite:
- Cross-attention head attribution across brain regions and frequency bands.
- Latent manifold vector field flow visualization (quiver diagrams of dz/dt vs z).
- UMAP / PCA manifold alignment tracking under electrode drift and session shift.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap

def extract_attention_rollout(cross_attn_weights):
    """
    cross_attn_weights: (B, H, T_spike, T_fmri)
    Returns:
        mean_attn: (T_spike, T_fmri) averaged across batches and heads
        head_diversity: (H, T_spike, T_fmri)
    """
    if isinstance(cross_attn_weights, torch.Tensor):
        cross_attn_weights = cross_attn_weights.detach().cpu().numpy()
    
    mean_attn = np.mean(cross_attn_weights, axis=(0, 1)) # (T_s, T_f)
    head_diversity = np.mean(cross_attn_weights, axis=0) # (H, T_s, T_f)
    return mean_attn, head_diversity


def plot_cross_attention_heatmap(mean_attn, segment_sec=8.0, save_path=None):
    """
    Plots the cross-modal attention heatmap between continuous spike time (ms)
    and fMRI BOLD acquisition time (sec), showing the biophysical HRF delay.
    """
    T_s, T_f = mean_attn.shape
    t_s = np.linspace(0, segment_sec, T_s)
    t_f = np.linspace(0, segment_sec, T_f)

    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(
        mean_attn,
        cmap="mako",
        cbar_kws={'label': 'Cross-Attention Weight'},
        xticklabels=np.round(t_f, 1),
        yticklabels=np.round(t_s[::max(1, T_s // 8)], 1)
    )
    plt.title("Cross-Modal Temporal Attention: Spike Queries vs. fMRI Keys/Values", fontsize=13, fontweight='bold')
    plt.xlabel("fMRI BOLD Acquisition Time (seconds)", fontsize=11)
    plt.ylabel("High-Density Spike Train Window Time (seconds)", fontsize=11)
    
    # Format tick marks
    ax.set_yticks(np.linspace(0, T_s - 1, 8))
    ax.set_yticklabels(np.round(np.linspace(0, segment_sec, 8), 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return plt.gcf()


def plot_latent_dynamics_phase_space(latent_traj, dz_traj, save_path=None):
    """
    Phase space vector field flow diagram (dz/dt vs z) showing continuous neural manifold trajectories.
    latent_traj: (T, d_z)
    dz_traj: (T, d_z)
    """
    if isinstance(latent_traj, torch.Tensor):
        latent_traj = latent_traj.detach().cpu().numpy()
    if isinstance(dz_traj, torch.Tensor):
        dz_traj = dz_traj.detach().cpu().numpy()

    z1 = latent_traj[:, 0]
    z2 = latent_traj[:, 1]
    dz1 = dz_traj[:, 0]
    dz2 = dz_traj[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Trajectory curve colored by time
    points = np.array([z1, z2]).T.reshape(-1, 1, 2)
    t = np.linspace(0, 1, len(z1))
    scatter = ax.scatter(z1, z2, c=t, cmap='plasma', s=16, alpha=0.8, zorder=2, label="Neural Trajectory")
    
    # Vector field flow quiver (subsampled for clarity)
    step = max(1, len(z1) // 30)
    ax.quiver(
        z1[::step], z2[::step],
        dz1[::step], dz2[::step],
        color='cyan', alpha=0.7, scale=40.0, width=0.004, zorder=3, label=r"Flow $\dot{\mathbf{z}}$"
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Normalized Trajectory Time", fontsize=10)
    ax.set_title("Latent Neural Manifold: Phase-Space Vector Field Flow", fontsize=13, fontweight='bold')
    ax.set_xlabel(r"Latent Factor $z_1$", fontsize=11)
    ax.set_ylabel(r"Latent Factor $z_2$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_manifold_alignment_comparison(source_latents, target_latents_unshifted, target_latents_adapted, save_path=None):
    """
    UMAP Manifold Projection comparing:
    - Session 0 (Source Baseline)
    - Session 1 (Target Drift: Unadapted) -> shows severe manifold distortion
    - Session 1 (Target Drift: Domain-Adapted) -> shows restored manifold alignment!
    """
    if isinstance(source_latents, torch.Tensor):
        source_latents = source_latents.detach().cpu().numpy()
    if isinstance(target_latents_unshifted, torch.Tensor):
        target_latents_unshifted = target_latents_unshifted.detach().cpu().numpy()
    if isinstance(target_latents_adapted, torch.Tensor):
        target_latents_adapted = target_latents_adapted.detach().cpu().numpy()

    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=15)
    
    # Fit UMAP on source baseline
    emb_source = reducer.fit_transform(source_latents)
    emb_unshifted = reducer.transform(target_latents_unshifted)
    emb_adapted = reducer.transform(target_latents_adapted)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Severe drift in unadapted model
    axes[0].scatter(emb_source[:, 0], emb_source[:, 1], c='#1f77b4', alpha=0.6, s=25, label="Source Session (Day 0)")
    axes[0].scatter(emb_unshifted[:, 0], emb_unshifted[:, 1], c='#d62728', alpha=0.6, s=25, label="Target Drift (Unadapted)")
    axes[0].set_title("Standard Decoder: Manifold Drift Failure\n(Electrode Drift + Unit Loss)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Subplot 2: Restored alignment under Domain Adaptation
    axes[1].scatter(emb_source[:, 0], emb_source[:, 1], c='#1f77b4', alpha=0.6, s=25, label="Source Session (Day 0)")
    axes[1].scatter(emb_adapted[:, 0], emb_adapted[:, 1], c='#2ca02c', alpha=0.6, s=25, label="Target Drift (Ours + DA)")
    axes[1].set_title("Ours (GRL Domain-Adapted):\nPreserved Latent Manifold Geometry", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
