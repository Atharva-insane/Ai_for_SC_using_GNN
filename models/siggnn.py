"""
SigGNN — Signature Graph Neural Network for Supply Chain Forecasting.

The master model that combines:
1. Multi-Scale Signature Encoder (temporal geometry)
2. Hierarchical Category Embeddings (cross-learning)
3. Sparse Temporal GAT (spatial dependencies)
4. Forecast Predictor MLP (28-day horizon output)
5. Hierarchical Reconciliation (coherent forecasts)

This architecture is designed to:
- Capture demand path geometry at multiple time scales
- Learn cross-item and cross-store dependencies
- Produce coherent hierarchical forecasts
- Support adversarial training for robustness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .signature import MultiScaleSignatureEncoder
from .gat import SparseTemporalGAT
from .reconciliation import SimpleReconciliation


class HierarchicalEmbeddings(nn.Module):
    """
    Learnable embeddings for categorical hierarchy levels.
    Maps integer IDs to dense vectors for:
    - store_id, dept_id, cat_id, state_id, item_id
    """

    def __init__(self, vocab_sizes: Dict[str, int], embed_dims: Dict[str, int]):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.embed_order = []

        for key in ['store_id', 'dept_id', 'cat_id', 'state_id', 'item_id']:
            vs_key = f'{key}_vocab_size'
            if vs_key in vocab_sizes:
                vocab_size = vocab_sizes[vs_key]
                embed_dim = embed_dims.get(key, 8)
                self.embeddings[key] = nn.Embedding(vocab_size, embed_dim)
                self.embed_order.append(key)

        self.output_dim = sum(
            self.embeddings[k].embedding_dim for k in self.embed_order
        )

    def forward(self, category_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            category_ids: dict of category_name → (N,) integer tensor
            
        Returns:
            (N, total_embed_dim) concatenated embeddings
        """
        embeds = []
        for key in self.embed_order:
            if key in category_ids:
                embeds.append(self.embeddings[key](category_ids[key]))

        return torch.cat(embeds, dim=-1)  # (N, total_embed_dim)


class ForecastPredictor(nn.Module):
    """
    Multi-horizon forecast predictor with residual connections.
    
    Outputs 28 daily predictions instead of a single mean.
    This allows WRMSSE computation per-day.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        horizon: int = 28,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.horizon = horizon

        layers = []
        current_dim = in_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final layer outputs `horizon` predictions
        layers.append(nn.Linear(current_dim, horizon))

        self.mlp = nn.Sequential(*layers)

        # Learnable day-of-horizon scaling
        # (accounts for weekend/weekday patterns within the 28-day window)
        self.horizon_scale = nn.Parameter(torch.ones(horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node representations
            
        Returns:
            (N, horizon) daily predictions
        """
        out = self.mlp(x)  # (N, horizon)
        out = out * F.softplus(self.horizon_scale)  # Scale per forecast day
        return out


class SigGNN(nn.Module):
    """
    The Signature-Graph Neural Network.
    
    Forward pass:
    1. Encode time series windows → multi-scale signatures
    2. Embed categorical hierarchy → dense vectors
    3. Concatenate signature + category features
    4. GNN message passing (spatial + hierarchical reasoning)
    5. Predict 28-day horizon
    6. Reconcile across hierarchy
    """

    def __init__(
        self,
        input_channels: int,
        vocab_sizes: Dict[str, int],
        sig_windows: list = [7, 28, 90],
        sig_depth: int = 3,
        use_lead_lag: bool = True,
        gat_hidden: int = 128,
        gat_heads: int = 4,
        gat_layers: int = 3,
        gat_edge_types: int = 3,
        predictor_hidden: int = 256,
        predictor_layers: int = 3,
        horizon: int = 28,
        dropout: float = 0.2,
        num_dept_groups: int = 7,
    ):
        super().__init__()

        # ── 1. Multi-Scale Signature Encoder ──
        self.sig_encoder = MultiScaleSignatureEncoder(
            input_channels=input_channels,
            windows=sig_windows,
            depth=sig_depth,
            use_lead_lag=use_lead_lag,
        )
        sig_dim = self.sig_encoder.get_output_dim()

        # ── 2. Hierarchical Embeddings ──
        embed_dims = {
            'store_id': 8,
            'dept_id': 8,
            'cat_id': 4,
            'state_id': 4,
            'item_id': 16,
        }
        self.hier_embed = HierarchicalEmbeddings(vocab_sizes, embed_dims)
        embed_dim = self.hier_embed.output_dim

        # ── 3. Feature fusion ──
        total_input_dim = sig_dim + embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, gat_hidden),
            nn.GELU(),
            nn.LayerNorm(gat_hidden),
        )

        # ── 4. Sparse Temporal GAT ──
        self.gat = SparseTemporalGAT(
            in_dim=gat_hidden,
            hidden_dim=gat_hidden,
            out_dim=gat_hidden,
            num_heads=gat_heads,
            num_layers=gat_layers,
            num_edge_types=gat_edge_types,
            dropout=dropout,
        )

        # ── 5. Forecast Predictor ──
        self.predictor = ForecastPredictor(
            in_dim=gat_hidden,
            hidden_dim=predictor_hidden,
            horizon=horizon,
            num_layers=predictor_layers,
            dropout=dropout + 0.1,  # Slightly more dropout in predictor
        )

        # ── 6. Reconciliation ──
        self.reconcile = SimpleReconciliation(
            num_groups=num_dept_groups,
            max_ratio=20.0,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        category_ids: Dict[str, torch.Tensor],
        dept_ids: Optional[torch.Tensor] = None,
        historical_mean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            node_features: (N, Seq_Len, C) per-node time series features
            edge_index: (2, E) graph edges
            edge_type: (E,) edge type labels
            category_ids: dict of category → (N,) integer IDs
            dept_ids: (N,) department group IDs for reconciliation
            historical_mean: (N,) mean daily sales for clipping
            
        Returns:
            (N, 28) daily demand forecasts
        """
        # 1. Multi-scale signatures
        sig_features = self.sig_encoder(node_features)  # (N, sig_dim)

        # 2. Category embeddings
        cat_features = self.hier_embed(category_ids)  # (N, embed_dim)

        # 3. Fuse
        h = torch.cat([sig_features, cat_features], dim=-1)  # (N, sig_dim + embed_dim)
        h = self.fusion(h)  # (N, gat_hidden)

        # 4. GNN message passing
        h = self.gat(h, edge_index, edge_type)  # (N, gat_hidden)

        # 5. Predict 28-day horizon
        predictions = self.predictor(h)  # (N, 28)

        # 6. Reconcile
        predictions = self.reconcile(
            predictions,
            group_ids=dept_ids,
            historical_mean=historical_mean,
        )

        return predictions

    def get_attention_weights(self) -> Dict:
        """Extract attention weights for visualization."""
        weights = {}
        for i, layer in enumerate(self.gat.gat_layers):
            weights[f'layer_{i}'] = {
                'attn_linear': layer.attn_linear.data.cpu(),
                'edge_type_embed': layer.edge_type_embed.weight.data.cpu(),
            }
        return weights

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TweedieLoss(nn.Module):
    """
    Tweedie deviance loss — the same loss used by the M5 1st place winner.
    
    The Tweedie distribution is ideal for zero-inflated continuous data
    (like retail sales). It naturally handles the large number of zeros.
    
    Deviance = 2 * [ y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p) ]
    
    For p=1.5 (compound Poisson-Gamma):
    Deviance = 2 * [ 2*sqrt(y) - y/sqrt(μ) - 2*sqrt(μ) + ... ]
    """

    def __init__(self, p: float = 1.5):
        """
        Args:
            p: Tweedie power parameter. Must be in (1, 2).
               p=1.0: Poisson
               p=1.5: Compound Poisson-Gamma (best for M5)
               p=2.0: Gamma
        """
        super().__init__()
        assert 1.0 < p < 2.0, f"Tweedie p must be in (1, 2), got {p}"
        self.p = p

    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N, horizon) predicted values (must be > 0)
            targets: (N, horizon) actual values (>= 0)
            
        Returns:
            Scalar loss
        """
        # Ensure predictions are positive
        mu = F.softplus(predictions) + 1e-8
        y = targets

        p = self.p

        # Tweedie deviance
        loss = -y * torch.pow(mu, 1 - p) / (1 - p) + \
               torch.pow(mu, 2 - p) / (2 - p)

        return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss that gives more weight to high-selling items.
    Approximates the WRMSSE weighting during training.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N, horizon) 
            targets: (N, horizon)
            weights: (N,) per-item weights (dollar-sales based)
        """
        se = (predictions - targets) ** 2  # (N, horizon)
        mse = se.mean(dim=1)  # (N,)

        if weights is not None:
            weights = weights / (weights.sum() + 1e-10)
            loss = (mse * weights).sum()
        else:
            loss = mse.mean()

        return loss
