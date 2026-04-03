"""
Hierarchical Forecast Reconciliation.

Ensures forecasts are coherent across the M5 hierarchy:
- Bottom-up: item-store → department → category → store → state → total
- Forecasts at aggregated levels should equal the sum of children

Uses a differentiable reconciliation layer inspired by MinT
(Minimum Trace optimal reconciliation, Wickramasuriya et al. 2019).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class HierarchicalReconciliation(nn.Module):
    """
    Differentiable hierarchical forecast reconciliation.
    
    Given raw bottom-level forecasts, this module:
    1. Computes aggregated forecasts at all levels
    2. Applies a learned reconciliation matrix to ensure coherence
    3. Returns adjusted bottom-level forecasts
    
    The reconciliation matrix is parameterized as:
    ŷ_adjusted = S × P × ŷ_raw
    
    where:
    - S is the summing matrix (maps bottom to all levels)
    - P is a learned projection matrix
    """

    def __init__(
        self,
        num_bottom: int,
        hierarchy_groups: Optional[Dict[str, List[List[int]]]] = None,
        method: str = 'bottom_up',  # 'bottom_up', 'learned', 'ols'
    ):
        """
        Args:
            num_bottom: number of bottom-level series
            hierarchy_groups: dict of level_name → list of groups (each group = list of indices)
            method: reconciliation method
        """
        super().__init__()
        self.num_bottom = num_bottom
        self.method = method

        if method == 'learned' and hierarchy_groups is not None:
            # Build the summing matrix S
            num_aggregated = sum(len(groups) for groups in hierarchy_groups.values())
            total = num_bottom + num_aggregated

            S = torch.zeros(total, num_bottom)
            # Bottom level: identity
            S[:num_bottom, :num_bottom] = torch.eye(num_bottom)
            # Aggregated levels
            idx = num_bottom
            for level_name, groups in hierarchy_groups.items():
                for group in groups:
                    for member in group:
                        if member < num_bottom:
                            S[idx, member] = 1.0
                    idx += 1

            self.register_buffer('S', S)

            # Learnable reconciliation weights
            self.P = nn.Linear(total, num_bottom, bias=False)
            nn.init.eye_(self.P.weight[:num_bottom, :num_bottom])
        else:
            self.S = None
            self.P = None

    def forward(
        self,
        base_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reconciliation to bottom-level forecasts.
        
        Args:
            base_forecasts: (N, horizon) raw bottom-level forecasts
            
        Returns:
            (N, horizon) reconciled forecasts
        """
        if self.method == 'bottom_up' or self.P is None:
            # Bottom-up: no adjustment needed for bottom level
            # Just ensure non-negativity
            return F.relu(base_forecasts)

        elif self.method == 'learned':
            # Stack bottom forecasts and aggregated
            # base_forecasts: (N, H)
            H = base_forecasts.size(1)

            # Compute aggregated forecasts for each horizon step
            reconciled = []
            for h in range(H):
                y_h = base_forecasts[:, h]  # (N,)

                # Compute all-level forecasts via summing matrix
                all_level = self.S @ y_h  # (total,)

                # Apply learned projection
                adjusted = self.P(all_level)  # (N,)

                reconciled.append(adjusted)

            return torch.stack(reconciled, dim=1)  # (N, H)

        return base_forecasts


# For the initial version, we use a simpler approach:
# enforce non-negativity and apply a learned scaling per department
class SimpleReconciliation(nn.Module):
    """
    Simple reconciliation layer that:
    1. Enforces non-negative forecasts (sales can't be negative)
    2. Applies learned per-group scaling factors
    3. Optionally clips extreme predictions
    """

    def __init__(self, num_groups: int = 7, max_ratio: float = 10.0):
        """
        Args:
            num_groups: number of department/category groups
            max_ratio: max ratio of prediction to historical mean
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_groups))
        self.bias = nn.Parameter(torch.zeros(num_groups))
        self.max_ratio = max_ratio

    def forward(
        self,
        predictions: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        historical_mean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N, horizon) raw predictions
            group_ids: (N,) department/category group index
            historical_mean: (N,) mean historical daily sales
        """
        # Non-negativity
        out = F.softplus(predictions)

        # Per-group scaling
        if group_ids is not None:
            scale = self.scale[group_ids].unsqueeze(1)  # (N, 1)
            bias = self.bias[group_ids].unsqueeze(1)    # (N, 1)
            out = out * F.softplus(scale) + bias

        # Clip extreme predictions
        if historical_mean is not None:
            upper = historical_mean.unsqueeze(1) * self.max_ratio
            out = torch.minimum(out, upper)

        return out


# Needed for the import
import torch.nn.functional as F
