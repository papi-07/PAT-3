"""
modules/module4_fusion.py
Autoencoder + Attention-based fusion of glycemic and tissue features (Module 4).
"""

import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────

class PhysiologicalAutoencoder(nn.Module):
    """
    Autoencoder that compresses the full physiological feature vector into a
    *latent_dim*-dimensional physiological stability vector.

    Encoder: input_dim → 64 → 32 → latent_dim
    Decoder: latent_dim → 32 → 64 → input_dim
    """

    def __init__(self, input_dim: int, latent_dim: int = cfg.AUTOENCODER_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """Returns (reconstruction, latent)."""
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────
# Attention-based fusion network
# ─────────────────────────────────────────────────────────────

class AttentionFusionNetwork(nn.Module):
    """
    Multi-head self-attention over the physiological stability vector,
    producing an attention-weighted fused feature vector.

    The latent vector is treated as a sequence of *num_heads* groups,
    each of dimension embed_dim//num_heads.
    """

    def __init__(
        self,
        latent_dim: int = cfg.AUTOENCODER_LATENT_DIM,
        num_heads: int  = cfg.ATTENTION_NUM_HEADS,
        out_dim: int    = None,
    ):
        super().__init__()
        # Ensure latent_dim is divisible by num_heads
        assert latent_dim % num_heads == 0, (
            f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.latent_dim = latent_dim
        self.num_heads  = num_heads
        self.out_dim    = out_dim or latent_dim

        self.attention  = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, self.out_dim)

    def forward(self, z: torch.Tensor):
        """
        z : (batch, latent_dim)
        Returns: (fused, attn_weights)
          fused       : (batch, out_dim)
          attn_weights: (batch, 1, 1)  – scalar attention for the single token
        """
        # Treat the latent vector as a sequence of length 1
        z_seq = z.unsqueeze(1)  # (batch, 1, latent_dim)
        attn_out, attn_weights = self.attention(z_seq, z_seq, z_seq)
        fused = self.norm(attn_out.squeeze(1) + z)  # residual
        fused = self.proj(fused)
        return fused, attn_weights


# ─────────────────────────────────────────────────────────────
# Combined fusion module
# ─────────────────────────────────────────────────────────────

class FeatureFusionModule:
    """
    Wrapper that:
      1. Trains an autoencoder on the combined feature matrix.
      2. Encodes features to the latent space.
      3. Applies attention-based fusion.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = cfg.AUTOENCODER_LATENT_DIM,
        num_heads:  int = cfg.ATTENTION_NUM_HEADS,
        device=cfg.DEVICE,
    ):
        self.device     = device
        self.latent_dim = latent_dim
        self.ae  = PhysiologicalAutoencoder(input_dim, latent_dim).to(device)
        self.attn = AttentionFusionNetwork(latent_dim, num_heads).to(device)
        self.input_mean = None
        self.input_std  = None

    def _normalise(self, X: np.ndarray) -> torch.Tensor:
        X_n = (X - self.input_mean) / (self.input_std + 1e-8)
        return torch.tensor(X_n, dtype=torch.float32, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = cfg.NUM_EPOCHS,
        batch_size: int = cfg.BATCH_SIZE,
        lr: float = cfg.LEARNING_RATE,
    ):
        """Train the autoencoder on X (N, input_dim)."""
        self.input_mean = X.mean(axis=0)
        self.input_std  = X.std(axis=0) + 1e-8

        X_t = self._normalise(X)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.ae.train()
        for _ in range(epochs):
            for (xb,) in loader:
                opt.zero_grad()
                recon, _ = self.ae(xb)
                loss = criterion(recon, xb)
                loss.backward()
                opt.step()

        self.ae.eval()
        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray):
        """
        Encode + attend X → fused features.

        Returns
        -------
        fused       : (N, out_dim) np.ndarray
        attn_weights: (N, 1, 1)   np.ndarray
        latent      : (N, latent_dim) np.ndarray
        """
        X_t = self._normalise(X)

        self.ae.eval()
        self.attn.eval()

        latent     = self.ae.encode(X_t)
        fused, attn = self.attn(latent)

        return (
            fused.cpu().numpy(),
            attn.cpu().numpy(),
            latent.cpu().numpy(),
        )

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> float:
        """MSE reconstruction error on X."""
        X_t = self._normalise(X)
        self.ae.eval()
        recon, _ = self.ae(X_t)
        return float(nn.MSELoss()(recon, X_t).item())
