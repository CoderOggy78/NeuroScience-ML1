#!/usr/bin/env python3
"""
download_real_data.py
Download authentic open-access benchmark data from:
1. Allen Brain Observatory (Neuropixels Visual Coding S3)
2. ABIDE Preprocessed fMRI (FCP-INDI AWS S3)
3. OpenNeuro ds002336 (Simultaneous EEG-fMRI BIDS)
"""

import os
import urllib.request
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ALLEN_DIR = os.path.join(DATA_DIR, "allen")
ABIDE_DIR = os.path.join(DATA_DIR, "abide")
OPENNEURO_DIR = os.path.join(DATA_DIR, "openneuro")

for d in [ALLEN_DIR, ABIDE_DIR, OPENNEURO_DIR]:
    os.makedirs(d, exist_ok=True)

DOWNLOADS = [
    # Allen Brain Observatory
    {
        "url": "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/sessions.csv",
        "dest": os.path.join(ALLEN_DIR, "sessions.csv"),
        "source": "Allen Brain Observatory (Sessions Table)"
    },
    {
        "url": "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/probes.csv",
        "dest": os.path.join(ALLEN_DIR, "probes.csv"),
        "source": "Allen Brain Observatory (Probes Table)"
    },
    {
        "url": "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/channels.csv",
        "dest": os.path.join(ALLEN_DIR, "channels.csv"),
        "source": "Allen Brain Observatory (Neuropixels Channels 3D CCF Coordinates)"
    },
    {
        "url": "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/session_715093703/session_715093703_analysis_metrics.csv",
        "dest": os.path.join(ALLEN_DIR, "session_715093703_analysis_metrics.csv"),
        "source": "Allen Brain Observatory (Session 715093703 Unit Analysis Metrics)"
    },
    # ABIDE Preprocessed fMRI
    {
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/Pitt_0050003_rois_aal.1D",
        "dest": os.path.join(ABIDE_DIR, "Pitt_0050003_rois_aal.1D"),
        "source": "ABIDE CPAC fMRI (Pitt 0050003 AAL 116 ROIs)"
    },
    {
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/NYU_0050952_rois_aal.1D",
        "dest": os.path.join(ABIDE_DIR, "NYU_0050952_rois_aal.1D"),
        "source": "ABIDE CPAC fMRI (NYU 0050952 AAL 116 ROIs)"
    },
    {
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/UM_1_0050272_rois_aal.1D",
        "dest": os.path.join(ABIDE_DIR, "UM_1_0050272_rois_aal.1D"),
        "source": "ABIDE CPAC fMRI (UM_1 0050272 AAL 116 ROIs)"
    },
    {
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/UCLA_1_0051201_rois_aal.1D",
        "dest": os.path.join(ABIDE_DIR, "UCLA_1_0051201_rois_aal.1D"),
        "source": "ABIDE CPAC fMRI (UCLA_1 0051201 AAL 116 ROIs)"
    },
    # OpenNeuro ds002336 (Simultaneous EEG-fMRI)
    {
        "url": "https://s3.amazonaws.com/openneuro.org/ds002336/dataset_description.json",
        "dest": os.path.join(OPENNEURO_DIR, "dataset_description.json"),
        "source": "OpenNeuro ds002336 (Dataset Description)"
    },
    {
        "url": "https://s3.amazonaws.com/openneuro.org/ds002336/participants.tsv",
        "dest": os.path.join(OPENNEURO_DIR, "participants.tsv"),
        "source": "OpenNeuro ds002336 (Participants TSV)"
    },
    {
        "url": "https://s3.amazonaws.com/openneuro.org/ds002336/sub-xp101/func/sub-xp101_task-eegfmriNF_bold.json",
        "dest": os.path.join(OPENNEURO_DIR, "sub-xp101_task-eegfmriNF_bold.json"),
        "source": "OpenNeuro ds002336 (Siemens 3T BOLD Acquisition Metadata)"
    },
    {
        "url": "https://s3.amazonaws.com/openneuro.org/ds002336/channels.json",
        "dest": os.path.join(OPENNEURO_DIR, "channels.json"),
        "source": "OpenNeuro ds002336 (EEG Channels Metadata)"
    }
]

def download_file(url, dest, source):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[OK] Already exists: {os.path.basename(dest)} ({os.path.getsize(dest)/1024:.1f} KB)")
        return
    print(f"Downloading {source} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest) / 1024
        print(f"  -> Saved {dest} ({sz:.1f} KB)")
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")

def main():
    print("=== Downloading Real Benchmark Datasets ===")
    for item in DOWNLOADS:
        download_file(item["url"], item["dest"], item["source"])
    print("\n=== Download Complete! ===")

if __name__ == "__main__":
    main()
