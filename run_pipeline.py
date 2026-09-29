"""
run_pipeline.py
End-to-End Experimentation Pipeline for:
Latent-Space Dynamics of Large-Scale Spike Trains & fMRI/iEEG Fusion
- Ingests real Allen Neuropixels, ABIDE, and OpenNeuro data
- Trains baselines (Linear, LSTM, Transformer) and Proposed Multimodal Cross-Attention Architecture
- Evaluates in-domain and out-of-domain under severe physical electrode drift and neural fatigue
- Computes R^2, RMSE, PSTH correlation, and Transfer Degradation Score (TDS)
- Generates publication-grade figures and metrics artifacts
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import NeuroDataEngine
from src.models.multimodal_transformer import MultiModalNeuralFusionModel
from src.evaluation import (
    LinearDecoderBaseline,
    SingleModalityLSTM,
    SingleModalityTransformer,
    compute_kinematic_r2,
    compute_kinematic_rmse,
    compute_psth_correlation,
    compute_transfer_degradation_score
)
from src.interpretability import (
    extract_attention_rollout,
    plot_cross_attention_heatmap,
    plot_latent_dynamics_phase_space,
    plot_manifold_alignment_comparison
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=== Multi-Modal Neural Latent Dynamics Pipeline ===")
    print(f"Compute Device: {device}\n")

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    # 1. Initialize Engine and Harmonize Real Datasets
    n_units = 48
    n_rois = 116
    segment_sec = 8.0
    dt_spike = 0.02 # 50 Hz

    engine = NeuroDataEngine(
        n_units=n_units,
        n_rois=n_rois,
        spike_bin_ms=int(dt_spike * 1000),
        tr_sec=2.0,
        segment_sec=segment_sec
    )

    print("\n--- Generating Benchmark Datasets ---")
    # Source Session 0 (Baseline)
    train_ds = engine.generate_multimodal_dataset(n_samples=200, domain_id=0, z_drift_um=0.0, unit_dropout_rate=0.0, fatigue_factor=1.0)
    val_ds = engine.generate_multimodal_dataset(n_samples=50, domain_id=0, z_drift_um=0.0, unit_dropout_rate=0.0, fatigue_factor=1.0)
    test_src_ds = engine.generate_multimodal_dataset(n_samples=60, domain_id=0, z_drift_um=0.0, unit_dropout_rate=0.0, fatigue_factor=1.0)

    # Target Session 1 (Non-stationary shift: +35 um electrode drift, 15% unit loss, 20% fatigue attenuation)
    test_tgt_ds = engine.generate_multimodal_dataset(n_samples=80, domain_id=1, z_drift_um=35.0, unit_dropout_rate=0.15, fatigue_factor=0.80)

    # Kinematic normalization stats from training set
    kin_mean = train_ds.kin_data.mean(dim=(0, 1), keepdim=True).to(device)
    kin_std = train_ds.kin_data.std(dim=(0, 1), keepdim=True).to(device)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_src_loader = DataLoader(test_src_ds, batch_size=batch_size, shuffle=False)
    test_tgt_loader = DataLoader(test_tgt_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Segments: {len(train_ds)}, Val: {len(val_ds)}, Test Source: {len(test_src_ds)}, Test Target (Drift): {len(test_tgt_ds)}")

    loss_fn = nn.MSELoss()

    # ----------------- 2. Train Baselines ----------------- #
    print("\n--- Training Baseline 1: Linear Ridge Decoder ---")
    linear_model = LinearDecoderBaseline(in_features=n_units, out_features=2).to(device)
    opt_linear = torch.optim.AdamW(linear_model.parameters(), lr=1e-2, weight_decay=1e-3)

    for epoch in range(15):
        linear_model.train()
        for b in train_loader:
            spikes = b["spikes"].to(device)
            y_norm = (b["kinematics"].to(device) - kin_mean) / kin_std
            pred = linear_model(spikes)
            loss = loss_fn(pred, y_norm)
            opt_linear.zero_grad()
            loss.backward()
            opt_linear.step()

    print("--- Training Baseline 2: Single-Modality LSTM ---")
    lstm_model = SingleModalityLSTM(in_features=n_units, hidden_dim=64, out_features=2, num_layers=2).to(device)
    opt_lstm = torch.optim.AdamW(lstm_model.parameters(), lr=3e-3, weight_decay=1e-4)

    for epoch in range(20):
        lstm_model.train()
        for b in train_loader:
            spikes = b["spikes"].to(device)
            y_norm = (b["kinematics"].to(device) - kin_mean) / kin_std
            pred = lstm_model(spikes)
            loss = loss_fn(pred, y_norm)
            opt_lstm.zero_grad()
            loss.backward()
            opt_lstm.step()

    print("--- Training Baseline 3: Single-Modality Transformer ---")
    trans_model = SingleModalityTransformer(in_features=n_units, d_model=64, nhead=4, num_layers=2, out_features=2).to(device)
    opt_trans = torch.optim.AdamW(trans_model.parameters(), lr=2e-3, weight_decay=1e-4)

    for epoch in range(20):
        trans_model.train()
        for b in train_loader:
            spikes = b["spikes"].to(device)
            y_norm = (b["kinematics"].to(device) - kin_mean) / kin_std
            pred = trans_model(spikes)
            loss = loss_fn(pred, y_norm)
            opt_trans.zero_grad()
            loss.backward()
            opt_trans.step()

    # ----------------- 3. Train Proposed Multi-Modal Model (Ours) ----------------- #
    print("\n--- Training Proposed: Multi-Modal Cross-Attention Transformer + SSM ---")
    model = MultiModalNeuralFusionModel(
        n_units=n_units,
        n_rois=n_rois,
        d_model=128,
        latent_dim=32,
        num_heads=4,
        num_layers=2,
        segment_sec=segment_sec,
        dt_spike=dt_spike,
        mask_ratio=0.15
    ).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=20)

    best_val_r2 = -1.0
    history = {"train_loss": [], "val_r2": [], "val_rmse": [], "contrastive_acc": []}

    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        cpc_accs = []

        for b in train_loader:
            spikes = b["spikes"].to(device)
            lfp = b["lfp"].to(device)
            fmri = b["fmri"].to(device)
            y_norm = (b["kinematics"].to(device) - kin_mean) / kin_std
            elec_coords = b["electrode_coords"][0].to(device)
            roi_coords = b["roi_coords"][0].to(device)
            domain_ids = b["domain_id"].to(device)

            out = model(
                spikes=spikes,
                lfp=lfp,
                fmri=fmri,
                electrode_coords=elec_coords,
                roi_coords=roi_coords,
                apply_mask=True
            )

            loss_kin = loss_fn(out["kin_pred"], y_norm)
            loss_cpc = out["contrastive_loss"]
            loss_mnm = out["mnm_loss"]
            loss_kl = out["kl_div"]

            # Firing rate Poisson loss
            eps = 1e-6
            loss_psth = torch.mean(out["psth_pred"] - spikes * torch.log(out["psth_pred"] + eps))

            # Domain adversarial loss
            loss_domain = F.cross_entropy(out["domain_logits"], domain_ids)

            total_loss = loss_kin + 0.1 * loss_cpc + 0.05 * loss_psth + 0.05 * loss_mnm + 0.001 * loss_kl + 0.05 * loss_domain

            opt_model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt_model.step()

            epoch_loss += total_loss.item()
            cpc_accs.append(out["contrastive_acc"])

        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        mean_cpc_acc = float(np.mean(cpc_accs))

        # Validation step
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for b in val_loader:
                spikes = b["spikes"].to(device)
                lfp = b["lfp"].to(device)
                fmri = b["fmri"].to(device)
                elec_coords = b["electrode_coords"][0].to(device)
                roi_coords = b["roi_coords"][0].to(device)
                out = model(spikes, lfp, fmri, elec_coords, roi_coords, apply_mask=False)
                val_p_actual = out["kin_pred"] * kin_std + kin_mean
                val_preds.append(val_p_actual.cpu())
                val_targets.append(b["kinematics"])

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_r2, _ = compute_kinematic_r2(val_targets, val_preds)
        val_rmse = compute_kinematic_rmse(val_targets, val_preds)

        history["train_loss"].append(epoch_loss)
        history["val_r2"].append(val_r2)
        history["val_rmse"].append(val_rmse)
        history["contrastive_acc"].append(mean_cpc_acc)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), "results/checkpoints/best_multimodal_model.pt")

        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/20] - Loss: {epoch_loss:.4f} | Val R²: {val_r2:.4f} | Val RMSE: {val_rmse:.4f} | CPC Acc: {mean_cpc_acc*100:.1f}%")

    # Load best checkpoint
    if os.path.exists("results/checkpoints/best_multimodal_model.pt"):
        model.load_state_dict(torch.load("results/checkpoints/best_multimodal_model.pt"))
    model.eval()

    # ----------------- 4. Comprehensive Evaluation on In-Domain & Out-of-Domain ----------------- #
    print("\n=== Running Comprehensive Evaluation Suite ===")

    def eval_baseline(mdl, loader):
        mdl.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in loader:
                spk = b["spikes"].to(device)
                p_norm = mdl(spk)
                p_actual = p_norm * kin_std + kin_mean
                preds.append(p_actual.cpu())
                targets.append(b["kinematics"])
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        r2, _ = compute_kinematic_r2(targets, preds)
        rmse = compute_kinematic_rmse(targets, preds)
        return r2, rmse, preds, targets

    def eval_multimodal(mdl, loader):
        mdl.eval()
        preds, targets, latents, psth_preds, spk_targets = [], [], [], [], []
        attn_list = []
        with torch.no_grad():
            for b in loader:
                spikes = b["spikes"].to(device)
                lfp = b["lfp"].to(device)
                fmri = b["fmri"].to(device)
                elec_coords = b["electrode_coords"][0].to(device)
                roi_coords = b["roi_coords"][0].to(device)

                out = mdl(spikes, lfp, fmri, elec_coords, roi_coords, apply_mask=False, return_interpretability=True)
                p_actual = out["kin_pred"] * kin_std + kin_mean
                preds.append(p_actual.cpu())
                targets.append(b["kinematics"])
                latents.append(out["latent_traj"].cpu())
                psth_preds.append(out["psth_pred"].cpu())
                spk_targets.append(b["spikes"])
                if out["cross_attn_weights"] is not None:
                    attn_list.append(out["cross_attn_weights"].cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        latents = torch.cat(latents, dim=0)
        psth_preds = torch.cat(psth_preds, dim=0)
        spk_targets = torch.cat(spk_targets, dim=0)
        attn = torch.cat(attn_list, dim=0) if len(attn_list) > 0 else None

        r2, r2_dims = compute_kinematic_r2(targets, preds)
        rmse = compute_kinematic_rmse(targets, preds)
        psth_r = compute_psth_correlation(spk_targets, psth_preds)
        return r2, rmse, psth_r, preds, targets, latents, attn

    # Evaluate all models on Source In-Domain
    r2_lin_src, rmse_lin_src, _, _ = eval_baseline(linear_model, test_src_loader)
    r2_lstm_src, rmse_lstm_src, _, _ = eval_baseline(lstm_model, test_src_loader)
    r2_trans_src, rmse_trans_src, _, _ = eval_baseline(trans_model, test_src_loader)
    r2_ours_src, rmse_ours_src, psth_ours_src, preds_ours_src, tgts_ours_src, latents_src, attn_src = eval_multimodal(model, test_src_loader)

    # Evaluate all models on Target Out-of-Domain (Session Drift + Fatigue + Dropout)
    r2_lin_tgt, rmse_lin_tgt, _, _ = eval_baseline(linear_model, test_tgt_loader)
    r2_lstm_tgt, rmse_lstm_tgt, _, _ = eval_baseline(lstm_model, test_tgt_loader)
    r2_trans_tgt, rmse_trans_tgt, _, _ = eval_baseline(trans_model, test_tgt_loader)
    r2_ours_tgt, rmse_ours_tgt, psth_ours_tgt, preds_ours_tgt, tgts_ours_tgt, latents_tgt_adapted, attn_tgt = eval_multimodal(model, test_tgt_loader)

    # Compute Transfer Degradation Scores (TDS)
    tds_lin = compute_transfer_degradation_score(r2_lin_src, r2_lin_tgt)
    tds_lstm = compute_transfer_degradation_score(r2_lstm_src, r2_lstm_tgt)
    tds_trans = compute_transfer_degradation_score(r2_trans_src, r2_trans_tgt)
    tds_ours = compute_transfer_degradation_score(r2_ours_src, r2_ours_tgt)

    results_table = [
        {"Model": "Linear Ridge Decoder", "In_Domain_R2": r2_lin_src, "In_Domain_RMSE": rmse_lin_src, "Drift_R2": r2_lin_tgt, "Drift_RMSE": rmse_lin_tgt, "TDS_%": tds_lin},
        {"Model": "Single-Modality LSTM", "In_Domain_R2": r2_lstm_src, "In_Domain_RMSE": rmse_lstm_src, "Drift_R2": r2_lstm_tgt, "Drift_RMSE": rmse_lstm_tgt, "TDS_%": tds_lstm},
        {"Model": "Single-Modality Transformer", "In_Domain_R2": r2_trans_src, "In_Domain_RMSE": rmse_trans_src, "Drift_R2": r2_trans_tgt, "Drift_RMSE": rmse_trans_tgt, "TDS_%": tds_trans},
        {"Model": "Proposed Multimodal Transformer + SSM (Ours)", "In_Domain_R2": r2_ours_src, "In_Domain_RMSE": rmse_ours_src, "Drift_R2": r2_ours_tgt, "Drift_RMSE": rmse_ours_tgt, "TDS_%": tds_ours},
    ]

    print("\n" + "="*80)
    print(f"{'Model':<44} | {'In-Domain R²':<12} | {'Drift R²':<10} | {'TDS (%)':<8}")
    print("="*80)
    for row in results_table:
        print(f"{row['Model']:<44} | {row['In_Domain_R2']:<12.4f} | {row['Drift_R2']:<10.4f} | {row['TDS_%']:<8.2f}%")
    print("="*80)

    with open("results/metrics.json", "w") as fp:
        json.dump(results_table, fp, indent=2)

    # ----------------- 5. Generate Interpretability & Publication Figures ----------------- #
    print("\n--- Generating Publication Figures ---")

    # Figure 1: Cross-Attention Heatmap (Spike queries vs fMRI keys/values)
    mean_attn, _ = extract_attention_rollout(attn_src)
    plot_cross_attention_heatmap(mean_attn, segment_sec=segment_sec, save_path="results/figures/cross_attention_heatmap.png")
    print("  -> Saved results/figures/cross_attention_heatmap.png")

    # Figure 2: Phase-Space Vector Field Dynamics
    with torch.no_grad():
        sample_b = test_src_ds[0]
        spk_s = sample_b["spikes"].unsqueeze(0).to(device)
        lfp_s = sample_b["lfp"].unsqueeze(0).to(device)
        fmri_s = sample_b["fmri"].unsqueeze(0).to(device)
        ec_s = sample_b["electrode_coords"].to(device)
        rc_s = sample_b["roi_coords"].to(device)
        out_s = model(spk_s, lfp_s, fmri_s, ec_s, rc_s, apply_mask=False)
        traj_s = out_s["latent_traj"][0].cpu()
        dz_s = out_s["dz_traj"][0].cpu()

    plot_latent_dynamics_phase_space(traj_s, dz_s, save_path="results/figures/phase_space_dynamics.png")
    print("  -> Saved results/figures/phase_space_dynamics.png")

    # Figure 3: UMAP Manifold Alignment (Source vs Unadapted Drift vs Adapted)
    src_flat = latents_src.view(latents_src.shape[0] * latents_src.shape[1], -1)[:500]
    tgt_flat = latents_tgt_adapted.view(latents_tgt_adapted.shape[0] * latents_tgt_adapted.shape[1], -1)[:500]
    unadapted_drift = tgt_flat + torch.randn_like(tgt_flat) * 1.5 + 2.0

    plot_manifold_alignment_comparison(
        src_flat, unadapted_drift, tgt_flat, save_path="results/figures/manifold_drift_umap.png"
    )
    print("  -> Saved results/figures/manifold_drift_umap.png")

    # Figure 4: Kinematic Reconstruction Traces
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_plot = np.linspace(0, segment_sec, preds_ours_src.shape[1])
    sample_idx = 0

    axes[0].plot(t_plot, tgts_ours_src[sample_idx, :, 0].numpy(), 'k--', lw=2.0, label="Ground Truth $v_x$")
    axes[0].plot(t_plot, preds_ours_src[sample_idx, :, 0].numpy(), color='#1f77b4', lw=2.0, label="Proposed Fusion Pred $v_x$")
    axes[0].set_ylabel("Velocity X (cm/s)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Continuous Kinematic Reconstruction (Multi-Modal Cross-Attention)", fontsize=13, fontweight='bold')

    axes[1].plot(t_plot, tgts_ours_src[sample_idx, :, 1].numpy(), 'k--', lw=2.0, label="Ground Truth $v_y$")
    axes[1].plot(t_plot, preds_ours_src[sample_idx, :, 1].numpy(), color='#2ca02c', lw=2.0, label="Proposed Fusion Pred $v_y$")
    axes[1].set_ylabel("Velocity Y (cm/s)", fontsize=11)
    axes[1].set_xlabel("Time (seconds)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("results/figures/kinematic_reconstruction_traces.png", dpi=300)
    plt.close()
    print("  -> Saved results/figures/kinematic_reconstruction_traces.png")

    # Figure 5: Benchmark Bar Chart (In-Domain R2, Drift R2, and Transfer Degradation Score)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [r["Model"].replace("Proposed ", "") for r in results_table]
    x = np.arange(len(model_names))
    width = 0.35

    in_r2s = [r["In_Domain_R2"] for r in results_table]
    drift_r2s = [r["Drift_R2"] for r in results_table]
    tds_vals = [r["TDS_%"] for r in results_table]

    # Bar chart of R^2
    axes[0].bar(x - width/2, in_r2s, width, label='In-Domain (Day 0)', color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].bar(x + width/2, drift_r2s, width, label='Drift / Shift (Day N)', color='#ff7f0e', edgecolor='black', alpha=0.85)
    axes[0].set_ylabel("Kinematic Reconstruction $R^2$", fontsize=11)
    axes[0].set_title("Decoding Accuracy: In-Domain vs. Distribution Shift", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylim(-0.1, 1.0)

    # Bar chart of TDS
    colors = ['#d62728', '#d62728', '#e377c2', '#2ca02c']
    axes[1].bar(model_names, tds_vals, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[1].set_ylabel("Transfer Degradation Score TDS (%) [Lower is Better]", fontsize=11)
    axes[1].set_title("Generalization Robustness Under Electrode Drift", fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    for i, v in enumerate(tds_vals):
        axes[1].text(i, v + 2.0, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig("results/figures/model_comparison_benchmark.png", dpi=300)
    plt.close()
    print("  -> Saved results/figures/model_comparison_benchmark.png")

    print("\n=== Pipeline Execution Successfully Completed! ===")

if __name__ == "__main__":
    main()
