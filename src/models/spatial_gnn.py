"""
spatial_gnn.py
Spatiotemporal Graph Neural Network and 3D Coordinate-Aware Neural Spatial Encoder.
Encodes physical electrode channel positions (Neuropixels CCF coordinates)
and fMRI anatomical ROI centroids (AAL MNI space) using distance-based graph message passing
and multi-query spatial attention pooling to preserve directional tuning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGraphConv(nn.Module):
    """
    Graph Convolutional Layer operating over spatial electrode/ROI coordinates:
    H^{(l+1)} = Activation( D^{-1/2} (A + I) D^{-1/2} H^{(l)} W )
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        """
        x: (B, T, N_nodes, in_features)
        adj: (N_nodes, N_nodes) normalized adjacency matrix
        """
        out = torch.matmul(adj, x) # (B, T, N, in_features)
        out = self.linear(out)
        out = F.gelu(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out

class CoordinateAwareSpatialEncoder(nn.Module):
    """
    Maps multi-channel neural activity (spikes/LFP or fMRI ROIs) into a geometry-aware
    spatiotemporal representation using normalized 3D physical coordinates and multi-query spatial attention.
    """
    def __init__(self, in_channels_per_node=2, hidden_dim=32, num_spatial_queries=4, out_dim=128, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.num_spatial_queries = num_spatial_queries
        self.out_dim = out_dim

        # 3D Coordinate Positional Embedding with LayerNorm
        self.coord_embed = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels_per_node, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Spatial Graph Message Passing
        self.gnn1 = SpatialGraphConv(hidden_dim, hidden_dim)
        self.gnn2 = SpatialGraphConv(hidden_dim, hidden_dim)

        # Multi-query spatial attention pooling:
        # Learns K distinct spatial receptive field readouts (e.g. directional tuning clusters)
        self.spatial_queries = nn.Parameter(torch.randn(num_spatial_queries, hidden_dim) * 0.05)
        
        # Final readout projection
        self.readout = nn.Sequential(
            nn.Linear(num_spatial_queries * hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def compute_adjacency(self, norm_coords):
        """
        Computes RBF kernel distance adjacency on normalized coordinates.
        norm_coords: (N_nodes, 3)
        """
        diff = norm_coords.unsqueeze(0) - norm_coords.unsqueeze(1) # (N, N, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1) # (N, N)
        adj = torch.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
        
        # Add self-loops
        adj = adj + torch.eye(norm_coords.size(0), device=norm_coords.device)
        
        # Degree normalization: D^{-1/2} A D^{-1/2}
        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return adj_norm

    def forward(self, x, coords):
        """
        x: (B, T, N_nodes, in_channels_per_node)
        coords: (N_nodes, 3) 3D coordinates (microns or mm)
        Returns:
            pooled: (B, T, out_dim) temporally aligned latent sequence
            h: (B, T, N_nodes, hidden_dim) node-level embeddings
        """
        B, T, N, _ = x.shape
        
        # Robust coordinate normalization to [-2, 2] standard space
        mean_c = coords.mean(dim=0, keepdim=True)
        std_c = coords.std(dim=0, keepdim=True) + 1e-5
        norm_coords = (coords - mean_c) / std_c

        adj = self.compute_adjacency(norm_coords) # (N, N)
        
        # Node activity projection + coordinate embedding
        h_node = self.node_proj(x) # (B, T, N, hidden_dim)
        h_coord = self.coord_embed(norm_coords).unsqueeze(0).unsqueeze(0) # (1, 1, N, hidden_dim)
        h = h_node + 0.1 * h_coord # Scale coordinate prior appropriately
        
        # Message passing layers
        h = self.gnn1(h, adj)
        h = self.gnn2(h, adj) # (B, T, N, hidden_dim)

        # Multi-query spatial attention pooling:
        # Q: (1, 1, K, hidden_dim)
        # K_mat: (B, T, N, hidden_dim)
        q = self.spatial_queries.view(1, 1, self.num_spatial_queries, self.hidden_dim)
        scores = torch.matmul(q, h.transpose(-2, -1)) / math.sqrt(self.hidden_dim) # (B, T, K, N)
        attn = torch.softmax(scores, dim=-1)
        
        # Pooled features: (B, T, K, hidden_dim)
        pooled = torch.matmul(attn, h)
        pooled = pooled.view(B, T, self.num_spatial_queries * self.hidden_dim)
        pooled = self.readout(pooled) # (B, T, out_dim)
        
        return pooled, h
