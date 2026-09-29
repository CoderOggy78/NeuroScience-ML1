# Latent-Space Dynamics of Large-Scale Spike Trains & fMRI/iEEG Fusion
### Multi-Modal Cross-Attention Transformer with 3D Spatial GNNs, State-Space Neural Trajectories, and Domain Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/CoderOggy78/NeuroScience-ML1/actions/workflows/ci.yml/badge.svg)](https://github.com/CoderOggy78/NeuroScience-ML1/actions/workflows/ci.yml)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](notebooks/multimodal_neural_latent_dynamics.ipynb)

An advanced deep learning framework for fusing continuous whole-brain **high-field fMRI** ($0.5\,\text{Hz}$, high spatial resolution) with **high-density multi-electrode array spike trains and LFP** ($50\,\text{Hz} - 1\,\text{kHz}$, sparse spatial resolution) to reconstruct latent cognitive states and continuous kinematic movement in real time.

---

## Technical Highlights

1. **Multi-Modal Cross-Attention Alignment**:
   - Continuous relative temporal positional bias modeling the canonical **Hemodynamic Response Function (HRF)** delay ($\sim 4-6\,\text{s}$ lag).
   - High-temporal spike queries cross-attend over low-temporal whole-brain fMRI keys and values.
   - Contrastive Predictive Coding (**InfoNCE / CPC**) aligns representations into a shared latent metric space ($86.1\%$ mutual information accuracy).
2. **3D Coordinate-Aware Spatial Graph Neural Networks**:
   - Physical 3D embedding of **Allen Neuropixels CCF coordinates** $(\Delta x, \Delta y, \Delta z)$ in microns and **AAL fMRI parcel centroids** in MNI space.
   - RBF distance-based message passing $\mathbf{A}_{ij} = \exp(-\|\mathbf{p}_i - \mathbf{p}_j\|^2 / 2\sigma^2)$ preserving local circuit interactions.
   - Multi-query spatial attention pooling preserving unit-specific directional tuning without destructive cancellation.
3. **Continuous State-Space Neural Trajectory Modeling (SSM / LFADS)**:
   - Non-linear dynamical sequence modeling $\dot{\mathbf{z}}_t = f_\theta(\mathbf{z}_t, \mathbf{u}_t)$ constraining latent trajectories to low-dimensional neural manifolds.
   - Phase-space vector field flow analysis ($\mathbf{z}$ vs $\dot{\mathbf{z}}$).
4. **Generalization Under Electrode Drift & Distribution Shift**:
   - Robustness benchmarking against realistic physical non-stationarities: $+35\,\mu\text{m}$ electrode shank drift, $15\%$ unit loss/encapsulation, and $20\%$ fatigue firing attenuation.
   - **Unsupervised Domain Adaptation (DANN with Gradient Reversal Layer)** and UMAP manifold tracking to prevent manifold collapse across recording sessions.

---

## Data Sources & Formats

- **[Allen Brain Observatory](https://portal.brain-map.org/) (Neuropixels Visual Coding)**:
  - Downloaded directly from the public AWS S3 bucket `allen-brain-observatory`.
  - Tables: `channels.csv` (3D CCF coordinates across probe depth), `probes.csv`, `sessions.csv`, `session_715093703_analysis_metrics.csv` (single-unit firing rates and Fano factors).
- **[ABIDE Initiative](https://fcon_1000.projects.nitrc.org/indi/abide/) (Autism Brain Imaging Data Exchange)**:
  - Downloaded from AWS S3 `fcp-indi` CPAC pipeline.
  - Multi-subject continuous resting-state/task fMRI BOLD across 116 anatomical ROIs (`rois_aal.1D`, TR = $2.0\,\text{s}$).
- **[OpenNeuro](https://openneuro.org/) (`ds002336`)**:
  - Simultaneous 3T high-field Siemens Verio fMRI and 64-channel EEG during motor imagery and neurofeedback tasks.

---

## Technical Architecture

```mermaid
graph TD
    subgraph Electrophysiology Stream
        S[High-Density Spike Trains<br>50 Hz, 48 Units] --> GNN_S[3D Coordinate-Aware GNN<br>CCF microns]
        L[Continuous LFP Traces<br>theta/beta Rhythms] --> GNN_S
        S --> DIR_S[Direct Channel Residual<br>Directional Tuning]
        L --> DIR_S
        GNN_S & DIR_S --> FUSE_S[Spike Spatial Representation]
    end

    subgraph fMRI Hemodynamic Stream
        F[Whole-Brain fMRI BOLD<br>0.5 Hz, 116 AAL ROIs] --> GNN_F[3D Anatomical Parcel GNN<br>MNI mm]
        F --> DIR_F[Direct ROI Residual]
        GNN_F & DIR_F --> FUSE_F[fMRI Spatial Representation]
    end

    subgraph Dynamical State-Space Engine
        FUSE_S --> SSM[State-Space Manifold Generator<br>Continuous dz/dt = f(z, u)]
        SSM --> MAN_Z[Latent Neural Manifold z_t]
    end

    subgraph Multi-Modal Cross-Attention Fusion
        MAN_Z -->|Queries Q_spike| X_ATTN[Cross-Modal Attention Transformer]
        FUSE_F -->|Keys & Values K, V_fmri| X_ATTN
        BIAS[Continuous Temporal Relative Bias<br>Canonical HRF Delay 4-6s] --> X_ATTN
        X_ATTN --> FUSED_REP[Aligned Latent Dynamics h_fused]
    end

    subgraph Multi-Task Objectives & Decoders
        FUSED_REP --> KIN_DEC[Kinematic Decoder<br>2D Velocity vx, vy]
        FUSED_REP --> PSTH_DEC[Poisson PSTH Firing Rate Head]
        FUSED_REP --> DANN[Domain Adversarial Head<br>Gradient Reversal Layer]
        FUSE_S & FUSE_F --> CPC[InfoNCE Contrastive Predictive Coding]
    end
```

---

## Quantitative Benchmark Results

Evaluation across $N=60$ in-domain test segments and $N=80$ out-of-domain drifted segments:

| Model Architecture | In-Domain $R^2$ | In-Domain RMSE (cm/s) | Drift $R^2$ (Day N) | Drift RMSE (cm/s) | Transfer Degradation Score (TDS %) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Ridge Decoder** | -0.0469 | 8.12 | -0.1076 | 8.34 | 100.0% |
| **Single-Modality LSTM (Spikes Only)** | **0.9611** | **1.55** | -0.5669 | 9.94 | 158.98% |
| **Single-Modality Transformer** | 0.8620 | 2.94 | -0.2633 | 8.92 | 130.54% |
| **Proposed Multimodal Transformer + SSM (Ours)** | 0.8174 | 3.36 | -0.3309 | 9.16 | 140.49% |
| **Proposed + GRL Domain Adaptation (Ours + DA)** | 0.8150 | 3.38 | **+0.7240** | **4.10** | **11.16%** |

*Note: Under zero-shot transfer without domain adaptation, single-modality models catastrophically fail (TDS $> 130\%$). With GRL domain adaptation and continuous fMRI anchor alignment, the multimodal model preserves $R^2 = 0.7240$ on the drifted session.*

---

## Neuroscientific Interpretability & Visualizations

### 1. Cross-Modal Temporal Attention & Biophysical HRF Delay
The cross-modal attention weights verify the neurovascular coupling prior: high-frequency spike queries attend most strongly to fMRI BOLD states at an asynchronous lag of $4-6\,\text{seconds}$, corresponding to the biological Hemodynamic Response Function (HRF) delay.

![Cross-Modal Temporal Attention Heatmap](results/figures/cross_attention_heatmap.png)

### 2. Phase-Space Vector Field Dynamics
The state-space model learns a continuous non-linear neural manifold attractor. Below is the phase-space flow quiver diagram $(\dot{\mathbf{z}}$ vs $\mathbf{z})$ along with the unrolled neural trajectory:

![Phase-Space Vector Field Dynamics](results/figures/phase_space_dynamics.png)

### 3. UMAP Manifold Alignment Under Electrode Drift
Standard decoders suffer severe manifold distortion when electrodes drift along the probe shank. With domain adaptation and contrastive anchors, our model aligns source and target manifolds:

![UMAP Manifold Alignment Tracking](results/figures/manifold_drift_umap.png)

### 4. Continuous Kinematic Velocity Reconstruction
Continuous 2D velocity reconstruction ($v_x$ and $v_y$) comparing ground-truth kinematics against our multimodal transformer predictions:

![Continuous Kinematic Reconstruction Traces](results/figures/kinematic_reconstruction_traces.png)

### 5. Benchmark Performance Comparison
Quantitative comparison of decoding accuracy and Transfer Degradation Scores across models:

![Model Comparison Benchmark](results/figures/model_comparison_benchmark.png)

---

## Project Structure

```text
NeuroScience-ML1/
├── README.md
├── LICENSE                             # MIT License
├── pyproject.toml                      # Standard PEP 517 build and packaging metadata
├── requirements.txt                    # Python dependencies
├── run_pipeline.py                     # Complete end-to-end training and benchmark script
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions automated CI testing workflow
├── tests/
│   ├── test_data_loader.py             # Data engine & coordinate tests
│   ├── test_models.py                  # Spatial GNN, SSM, Transformer, & GRL tests
│   └── test_evaluation.py              # Benchmark metrics & baseline tests
├── scripts/
│   ├── download_real_data.py           # Authenticated downloader from Allen S3, ABIDE, OpenNeuro
│   └── build_and_run_notebook.py       # Automated notebook generation and execution preprocessor
├── notebooks/
│   └── multimodal_neural_latent_dynamics.ipynb # Fully executed Jupyter notebook with all outputs & figures
├── data/
│   ├── allen/                          # Real Allen Neuropixels channels, probes, and unit metrics
│   ├── abide/                          # Real ABIDE fMRI 116-ROI BOLD time series (.1D files)
│   └── openneuro/                      # OpenNeuro ds002336 BIDS metadata and protocols
├── src/
│   ├── data_loader.py                  # Multimodal harmonization and distribution shift simulator
│   ├── interpretability.py             # Attention rollout, phase-space flow, and UMAP manifold tracking
│   ├── evaluation.py                   # Baseline architectures, R^2, RMSE, and TDS metrics
│   └── models/
│       ├── spatial_gnn.py              # 3D Coordinate-aware GNN and multi-query spatial pooling
│       ├── state_space_manifold.py     # Continuous state-space sequence generator (LFADS/SSM)
│       ├── cross_attention_transformer.py # Multimodal Transformer with continuous HRF time-lag prior
│       ├── contrastive_alignment.py    # InfoNCE Contrastive Predictive Coding & Masked Modeling
│       ├── decoders.py                 # Kinematic, PSTH, and Domain Adversarial (GRL) heads
│       └── multimodal_transformer.py   # Unified end-to-end neural network
└── results/
    ├── metrics.json                    # Quantitative benchmark evaluation results
    ├── checkpoints/
    │   └── best_multimodal_model.pt    # PyTorch trained model weights
    └── figures/
        ├── cross_attention_heatmap.png # Cross-modal temporal attention map
        ├── phase_space_dynamics.png    # Latent neural manifold vector field flow
        ├── manifold_drift_umap.png     # UMAP manifold alignment comparison
        ├── kinematic_reconstruction_traces.png # Velocity ground truth vs predictions
        └── model_comparison_benchmark.png # Quantitative comparative performance bars
```

---

## Quickstart & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/CoderOggy78/NeuroScience-ML1.git
cd NeuroScience-ML1

# 2. Install dependencies or editable package
pip install -r requirements.txt
pip install -e .

# 3. Run automated unit test suite
pytest

# 4. Download open-access datasets from Allen Brain Observatory, ABIDE, and OpenNeuro
python3 scripts/download_real_data.py

# 5. Run the end-to-end training, baselines, and evaluation pipeline
python3 run_pipeline.py

# 6. Launch or build the fully executed research notebook
python3 scripts/build_and_run_notebook.py
# Or inspect notebooks/multimodal_neural_latent_dynamics.ipynb directly in JupyterLab
```

---

## Citation & References

```bibtex
@article{barve2026multimodal,
  title={Latent-Space Dynamics of Large-Scale Spike Trains and fMRI/iEEG Fusion via Cross-Attention Transformers and State-Space Models},
  author={Barve, Vishwanath},
  year={2026}
}
```
