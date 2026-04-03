"""
Multi-Scale Signature Encoder with Lead-Lag Augmentation.

This module implements the core mathematical innovation of SigGNN:
path signatures at multiple temporal resolutions, capturing the
geometry of demand streams at different scales.

Mathematical background:
- The path signature is an infinite series of iterated integrals
  that provides a universal and faithful descriptor of a path.
- Truncated at depth m, it captures all cross-correlations
  between channels up to order m.
- The lead-lag augmentation captures the quadratic variation,
  making it sensitive to the "roughness" of price/demand paths.

References:
- Chevyrev & Kormilitzin (2016): "A Primer on the Signature Method"
- Kidger & Lyons (2020): "Signatory: Differentiable computations of
  the signature and logsignature transforms..."
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
# Signature computation — handles the signatory import gracefully
# ═══════════════════════════════════════════════════════════════

try:
    from signatory import signature as sig_compute
    from signatory import signature_channels as sig_channels
    from signatory import logsignature as logsig_compute
    from signatory import logsignature_channels as logsig_channels
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    print("⚠️ signatory not found. Using manual signature approximation.")


def manual_signature_depth2(path: torch.Tensor) -> torch.Tensor:
    """
    Manual computation of the depth-2 truncated signature.
    Fallback when signatory is not available.
    
    For a path X: R → R^d, the depth-2 signature includes:
    - Level 1: ∫ dX_i (increments)
    - Level 2: ∫∫ dX_i ⊗ dX_j (cross-integrals / area elements)
    
    Args:
        path: (batch, length, channels) tensor
        
    Returns:
        (batch, d + d²) signature tensor
    """
    B, L, d = path.shape
    increments = path[:, 1:, :] - path[:, :-1, :]  # (B, L-1, d)

    # Level 1: sum of increments (= endpoint - startpoint)
    level1 = increments.sum(dim=1)  # (B, d)

    # Level 2: ∫∫ dX_i ⊗ dX_j
    # Computed as: sum_{s<t} (X_{s+1} - X_s) ⊗ (X_{t+1} - X_t)
    cumsum = torch.cumsum(increments, dim=1)  # (B, L-1, d)
    # For each timestep t, compute increment_t ⊗ cumsum_{t-1}
    prev_cumsum = torch.cat([
        torch.zeros(B, 1, d, device=path.device),
        cumsum[:, :-1, :]
    ], dim=1)  # (B, L-1, d)

    # Outer product and sum
    level2 = torch.einsum('bti,btj->bij', increments, prev_cumsum)  # (B, d, d)
    level2 = level2.reshape(B, d * d)

    return torch.cat([level1, level2], dim=-1)  # (B, d + d²)


def compute_signature(path: torch.Tensor, depth: int, 
                      use_logsig: bool = False) -> torch.Tensor:
    """
    Compute the (log-)signature of a path, with fallback.
    
    Args:
        path: (batch, length, channels)
        depth: truncation depth
        use_logsig: if True, compute log-signature (more compact)
        
    Returns:
        (batch, sig_dim) signature tensor
    """
    if HAS_SIGNATORY:
        if use_logsig:
            return logsig_compute(path, depth, basepoint=True)
        return sig_compute(path, depth, basepoint=True)
    else:
        # Manual fallback (depth-2 only)
        if depth > 2:
            print("⚠️ Manual signature only supports depth 2. Using depth=2.")
        return manual_signature_depth2(path)


def get_signature_dim(channels: int, depth: int, 
                      use_logsig: bool = False) -> int:
    """Get the output dimension of the signature."""
    if HAS_SIGNATORY:
        if use_logsig:
            return logsig_channels(channels, depth)
        return sig_channels(channels, depth)
    else:
        # Manual depth-2: d + d²
        return channels + channels * channels


# ═══════════════════════════════════════════════════════════════
# Lead-Lag Augmentation
# ═══════════════════════════════════════════════════════════════

class LeadLagAugmentation(nn.Module):
    """
    Lead-lag transformation of a path.
    
    Given a path X = (X_1, ..., X_n), the lead-lag path is:
    (X_1, X_1), (X_1, X_2), (X_2, X_2), (X_2, X_3), ...
    
    This captures the quadratic variation of the path, which
    is crucial for distinguishing paths with the same increments
    but different volatility patterns.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, channels)
            
        Returns:
            (batch, 2*length - 1, 2*channels) lead-lag augmented path
        """
        B, L, C = x.shape
        # Lead: (X_1, X_2, X_2, X_3, X_3, ...)
        # Lag:  (X_1, X_1, X_2, X_2, X_3, ...)
        lead = torch.repeat_interleave(x, 2, dim=1)[:, 1:, :]   # (B, 2L-1, C)
        lag = torch.repeat_interleave(x, 2, dim=1)[:, :-1, :]    # (B, 2L-1, C)
        return torch.cat([lead, lag], dim=-1)  # (B, 2L-1, 2C)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale Signature Encoder
# ═══════════════════════════════════════════════════════════════

class MultiScaleSignatureEncoder(nn.Module):
    """
    Computes path signatures at multiple temporal resolutions.
    
    Architecture:
    1. Extract windows of different sizes from the end of the time series
    2. Optionally apply lead-lag augmentation
    3. Apply a learnable projection to each window (stream-preserving)
    4. Compute truncated path signatures
    5. Concatenate multi-scale signatures
    
    This captures:
    - Short window (7d): weekly patterns, recent trend
    - Medium window (28d): monthly patterns, price effects
    - Long window (90d): seasonal patterns, structural changes
    """

    def __init__(
        self,
        input_channels: int,
        windows: List[int] = [7, 28, 90],
        depth: int = 3,
        use_lead_lag: bool = True,
        use_logsig: bool = False,
        projection_dim: Optional[int] = None,
    ):
        """
        Args:
            input_channels: number of features per timestep
            windows: list of window sizes for multi-scale
            depth: signature truncation depth
            use_lead_lag: whether to apply lead-lag transform
            use_logsig: whether to use log-signature (more compact)
            projection_dim: if set, project input channels before signature
        """
        super().__init__()
        self.windows = sorted(windows)
        self.depth = depth
        self.use_lead_lag = use_lead_lag
        self.use_logsig = use_logsig

        # Optional learnable projection (stream-preserving)
        self.projection_dim = projection_dim or input_channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_channels, self.projection_dim),
                nn.GELU(),
                nn.Linear(self.projection_dim, self.projection_dim),
            )
            for _ in windows
        ])

        # Lead-lag augmentation
        self.lead_lag = LeadLagAugmentation() if use_lead_lag else nn.Identity()

        # Compute signature dimensions
        sig_input_dim = self.projection_dim * 2 if use_lead_lag else self.projection_dim
        self.sig_dim_per_window = get_signature_dim(sig_input_dim, depth, use_logsig)
        self.output_dim = self.sig_dim_per_window * len(windows)

        # Optional layer norm for each scale
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.sig_dim_per_window)
            for _ in windows
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_channels) — full time series features
            
        Returns:
            (batch, output_dim) — concatenated multi-scale signatures
        """
        B, T, C = x.shape
        sig_list = []

        for i, w in enumerate(self.windows):
            # Extract the last `w` timesteps (or all if shorter)
            actual_w = min(w, T)
            window = x[:, -actual_w:, :]  # (B, w, C)

            # Ensure minimum sequence length for signature (need ≥ 2)
            if actual_w < 2:
                # Pad with first value
                pad = window[:, :1, :].expand(B, 2 - actual_w, C)
                window = torch.cat([pad, window], dim=1)

            # Project
            window = self.projections[i](window)  # (B, w, proj_dim)

            # Lead-lag augmentation
            window = self.lead_lag(window)  # (B, 2w-1, 2*proj_dim) or (B, w, proj_dim)

            # Compute signature
            sig = compute_signature(window, self.depth, self.use_logsig)  # (B, sig_dim)

            # Normalize
            sig = self.norms[i](sig)

            sig_list.append(sig)

        # Concatenate all scales
        return torch.cat(sig_list, dim=-1)  # (B, total_sig_dim)

    def get_output_dim(self) -> int:
        """Return the total output dimension."""
        return self.output_dim
