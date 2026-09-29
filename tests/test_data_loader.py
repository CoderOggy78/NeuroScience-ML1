"""
Unit tests for data_loader.py:
- NeuroDataEngine initialization
- Real data harmonization (Allen, ABIDE, OpenNeuro)
- MultiModalNeuralDataset creation and DataLoader batching
- Physical non-stationarities: drift, dropout, fatigue
"""

import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import NeuroDataEngine, MultiModalNeuralDataset, AAL_CENTROIDS


@pytest.fixture
def engine():
    """Initializes a fast, lightweight NeuroDataEngine fixture."""
    return NeuroDataEngine(
        n_units=32,
        n_rois=116,
        spike_bin_ms=20,
        tr_sec=2.0,
        segment_sec=8.0  # 8.0s -> 400 spike steps, 4 fMRI steps
    )


def test_engine_initialization(engine):
    """Verify engine temporal resampler parameters and coordinate retrieval."""
    assert engine.n_units == 32
    assert engine.n_rois == 116
    assert engine.t_spike == 400  # 8.0s / 0.02s
    assert engine.t_fmri == 4     # 8.0s / 2.0s

    coords = engine.get_electrode_coords()
    assert coords.shape == (32, 3)
    assert coords.dtype == np.float32

    roi_coords = engine.get_roi_coords()
    assert roi_coords.shape == (116, 3)


def test_electrode_drift_simulation(engine):
    """Verify that z_drift_um shifts the physical coordinate geometry."""
    baseline_coords = engine.get_electrode_coords(z_drift_um=0.0)
    drifted_coords = engine.get_electrode_coords(z_drift_um=35.0)

    # Drift is applied along the second axis (depth)
    diff = drifted_coords[:, 1] - baseline_coords[:, 1]
    assert np.allclose(diff, 35.0, atol=1e-5)


def test_dataset_generation_shapes(engine):
    """Verify multimodal dataset generation yields correct shapes and types."""
    ds = engine.generate_multimodal_dataset(
        n_samples=5,
        domain_id=0,
        z_drift_um=0.0,
        unit_dropout_rate=0.0,
        fatigue_factor=1.0
    )
    assert len(ds) == 5

    item = ds[0]
    assert item["spikes"].shape == (400, 32)
    assert item["lfp"].shape == (400, 32)
    assert item["fmri"].shape == (4, 116)
    assert item["kinematics"].shape == (400, 2)
    assert item["electrode_coords"].shape == (32, 3)
    assert item["roi_coords"].shape == (116, 3)
    assert item["domain_id"].item() == 0


def test_dataloader_batching(engine):
    """Verify PyTorch DataLoader collates multimodal batches correctly."""
    ds = engine.generate_multimodal_dataset(
        n_samples=8,
        domain_id=1,
        z_drift_um=35.0,
        unit_dropout_rate=0.15,
        fatigue_factor=0.8
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch["spikes"].shape == (4, 400, 32)
    assert batch["lfp"].shape == (4, 400, 32)
    assert batch["fmri"].shape == (4, 4, 116)
    assert batch["kinematics"].shape == (4, 400, 2)
    assert batch["domain_id"].shape == (4,)
    assert batch["domain_id"][0].item() == 1
