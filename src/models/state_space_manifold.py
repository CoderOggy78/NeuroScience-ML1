"""
state_space_manifold.py
Vectorized Latent Neural Trajectory Modeling via State-Space Dynamical Systems (SSM / LFADS-inspired Generator).
Models non-linear hidden neural manifold dynamics z_t = f(z_{t-1}, u_t)
with continuous vector fields and dynamical state-space regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralVectorField(nn.Module):
    """
    Non-linear dynamical vector field f_theta(z, u) parameterized by a multi-layer perceptron.
    dz/dt = f_theta(z, u)
    """
    def __init__(self, latent_dim=32, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, u):
        """
        z: (B, T, latent_dim)
        u: (B, T, input_dim)
        Returns:
            dz: (B, T, latent_dim)
        """
        inp = torch.cat([z, u], dim=-1)
        dz = self.net(inp)
        return dz


class StateSpaceDynamicalModel(nn.Module):
    """
    Vectorized Continuous-Time State-Space Dynamical Model for Neural Manifolds.
    Encodes neural activity sequences into initial latent conditions z_0,
    then unrolls dynamical vector field trajectories z_t over time using vectorized state-space transitions.
    """
    def __init__(self, input_dim=128, latent_dim=32, controller_dim=32, dt=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.controller_dim = controller_dim
        self.dt = dt

        # Bidirectional Inference Network to infer initial latent state z_0
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=controller_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc_z0_mean = nn.Linear(controller_dim * 2, latent_dim)
        self.fc_z0_logvar = nn.Linear(controller_dim * 2, latent_dim)

        # Vectorized Latent Generator GRU
        self.generator_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=latent_dim,
            batch_first=True
        )

        # Vector field evaluator: dz/dt = f(z_t, x_t)
        self.vector_field = NeuralVectorField(
            latent_dim=latent_dim,
            input_dim=input_dim,
            hidden_dim=64
        )

        # Output factor projection
        self.factor_proj = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def encode_initial_state(self, x):
        """
        x: (B, T, input_dim)
        Returns: z0_mean, z0_logvar
        """
        _, h_n = self.encoder_rnn(x) # h_n: (2, B, controller_dim)
        h_bidir = torch.cat([h_n[0], h_n[1]], dim=-1) # (B, controller_dim * 2)
        mean = self.fc_z0_mean(h_bidir)
        logvar = self.fc_z0_logvar(h_bidir)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 5.0))
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x, return_dynamics=False):
        """
        x: (B, T, input_dim)
        Returns:
            factors: (B, T, input_dim)
            latent_traj: (B, T, latent_dim)
            dict with kl_loss, vector_field dz/dt
        """
        B, T, _ = x.shape
        z0_mean, z0_logvar = self.encode_initial_state(x)
        z_0 = self.reparameterize(z0_mean, z0_logvar) # (B, latent_dim)

        # Vectorized recurrent manifold generator initialized from z_0
        h_0 = z_0.unsqueeze(0) # (1, B, latent_dim)
        latent_traj, _ = self.generator_gru(x, h_0) # (B, T, latent_dim)

        # Compute continuous vector field dz/dt
        dz_traj = self.vector_field(latent_traj, x) # (B, T, latent_dim)

        # Output factor projection
        factors = self.factor_proj(latent_traj) # (B, T, input_dim)

        # KL Divergence against standard Gaussian prior N(0, I)
        kl_div = -0.5 * torch.sum(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp(), dim=-1).mean()

        info = {
            "kl_div": kl_div,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
            "dz_traj": dz_traj
        }
        if return_dynamics:
            return factors, latent_traj, info
        return factors, latent_traj
