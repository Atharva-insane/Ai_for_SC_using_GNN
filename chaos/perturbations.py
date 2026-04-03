"""
Chaos Engineering — Perturbation Strategies for Supply Chain Robustness.

Implements 6 supply chain disruption scenarios:
1. Demand Shock: Sudden demand spike or crash (COVID, viral product)
2. Supply Disruption: Stockout, shipment delay (zero sales windows)
3. Price Volatility: Competitor price war, inflation
4. Calendar Shift: Holiday moved, unexpected event
5. Graph Corruption: Store closure, supply chain reconfiguration
6. Adversarial Attack: Worst-case gradient-based perturbation (FGSM/PGD)

Each perturbation is designed to be:
- Differentiable (for adversarial training)
- Configurable (severity, duration, scope)
- Reproducible (seeded random state)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class Perturbation(ABC):
    """Base class for all perturbations."""

    def __init__(self, severity: float = 0.5, seed: Optional[int] = None):
        self.severity = severity
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply perturbation and return modified inputs.
        
        Returns:
            (node_features, edge_index, edge_type) — modified versions
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(severity={self.severity})"


class DemandShock(Perturbation):
    """
    Simulates sudden demand spikes or crashes.
    
    Real-world analogues:
    - COVID lockdown (demand crash for in-store items)
    - Viral TikTok product (10× demand spike)
    - Natural disaster (regional demand collapse)
    
    Implementation:
    - Select random items and time windows
    - Multiply demand by shock_factor (e.g., 0.1× for crash, 10× for spike)
    - Affect only the demand-related features (log-demand, lags, rolling stats)
    """

    def __init__(
        self,
        severity: float = 0.5,
        window_size: int = 14,
        item_fraction: float = 0.2,
        shock_type: str = 'mixed',  # 'spike', 'crash', 'mixed'
        seed: Optional[int] = None,
    ):
        super().__init__(severity, seed)
        self.window_size = window_size
        self.item_fraction = item_fraction
        self.shock_type = shock_type

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        # Select random items to shock
        num_shocked = max(1, int(N * self.item_fraction))
        shocked_items = self.rng.choice(N, num_shocked, replace=False)

        # Select random window within the time series
        max_start = max(0, T - self.window_size)
        window_start = self.rng.randint(0, max_start + 1)
        window_end = min(window_start + self.window_size, T)

        # Compute shock factors
        for item in shocked_items:
            if self.shock_type == 'spike':
                factor = 1.0 + self.severity * (5.0 + self.rng.exponential(3.0))
            elif self.shock_type == 'crash':
                factor = max(0.01, 1.0 - self.severity * self.rng.uniform(0.5, 1.0))
            else:  # mixed
                if self.rng.random() > 0.5:
                    factor = 1.0 + self.severity * (5.0 + self.rng.exponential(3.0))
                else:
                    factor = max(0.01, 1.0 - self.severity * self.rng.uniform(0.5, 1.0))

            # Apply to demand feature (channel 0 = log1p(demand))
            features[item, window_start:window_end, 0] = (
                torch.expm1(features[item, window_start:window_end, 0]) * factor
            )
            features[item, window_start:window_end, 0] = torch.log1p(
                F.relu(features[item, window_start:window_end, 0])
            )

        return features, edge_index, edge_type


class SupplyDisruption(Perturbation):
    """
    Simulates stockouts and supply chain disruptions.
    
    Real-world analogues:
    - Warehouse closure
    - Shipping container delay
    - Supplier factory shutdown
    
    Implementation:
    - Zero out demand for random items during random windows
    - This mimics "stockout" — item was available but not sold
    """

    def __init__(
        self,
        severity: float = 0.5,
        window_size: int = 14,
        item_fraction: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(severity, seed)
        self.window_size = window_size
        self.item_fraction = item_fraction

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        num_disrupted = max(1, int(N * self.item_fraction * self.severity))
        disrupted_items = self.rng.choice(N, num_disrupted, replace=False)

        win = int(self.window_size * self.severity)
        for item in disrupted_items:
            start = self.rng.randint(0, max(1, T - win))
            end = min(start + win, T)
            # Zero out demand and demand-derived features (first few channels)
            features[item, start:end, 0] = 0.0  # log1p(demand) = 0

        return features, edge_index, edge_type


class PriceVolatility(Perturbation):
    """
    Simulates price instability.
    
    Real-world analogues:
    - Competitor price war
    - Supply-side inflation
    - Promotional pricing errors
    
    Implementation:
    - Add heavy-tailed noise to price features
    - Use Student-t distribution for realistic fat tails
    """

    def __init__(
        self,
        severity: float = 0.3,
        seed: Optional[int] = None,
    ):
        super().__init__(severity, seed)

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        price_channel_idx: int = -6,  # Depends on feature ordering
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        # Generate heavy-tailed noise (Student-t with df=3)
        noise = torch.from_numpy(
            self.rng.standard_t(3, size=(N, T))
        ).float().to(features.device)

        # Scale by severity
        noise = 1.0 + self.severity * noise * 0.3

        # Apply to price-related channels
        # Price features are typically at fixed channel indices
        # After: [demand(1), lags(6), rolling(8), price(3), calendar(8)]
        # Price channels start at index 15 (1+6+8 = 15)
        price_start = 15  # Adjust based on actual feature layout
        price_end = min(price_start + 3, C)

        if price_start < C:
            for ch in range(price_start, price_end):
                features[:, :, ch] = features[:, :, ch] * noise

        return features, edge_index, edge_type


class CalendarShift(Perturbation):
    """
    Simulates calendar anomalies.
    
    Real-world analogues:
    - Holiday postponement
    - Unexpected store closure
    - Regional event changes
    
    Implementation:
    - Circular shift calendar features by ±k days
    - Randomly flip event indicators
    """

    def __init__(
        self,
        severity: float = 0.5,
        max_shift: int = 3,
        seed: Optional[int] = None,
    ):
        super().__init__(severity, seed)
        self.max_shift = max_shift

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, C = node_features.shape
        features = node_features.clone()

        # Calendar features are the last 8 channels
        cal_start = C - 8

        # Random shift
        shift = self.rng.randint(-self.max_shift, self.max_shift + 1)
        if shift != 0 and cal_start < C:
            features[:, :, cal_start:] = torch.roll(
                features[:, :, cal_start:], shift, dims=1
            )

        return features, edge_index, edge_type


class GraphCorruption(Perturbation):
    """
    Simulates supply chain graph disruptions.
    
    Real-world analogues:
    - Store closure (remove all edges to a store)
    - New competitor (add noise edges)
    - Supply chain restructuring (shuffle edges)
    
    Implementation:
    - Randomly drop edges from the graph
    - Optionally add random false edges
    """

    def __init__(
        self,
        severity: float = 0.2,
        drop_ratio: float = 0.2,
        add_noise_edges: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(severity, seed)
        self.drop_ratio = drop_ratio
        self.add_noise_edges = add_noise_edges

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = edge_index.size(1)
        N = node_features.size(0)

        # Drop random edges
        num_drop = int(E * self.drop_ratio * self.severity)
        keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)

        if num_drop > 0 and num_drop < E:
            drop_indices = self.rng.choice(E, num_drop, replace=False)
            keep_mask[drop_indices] = False

        new_edge_index = edge_index[:, keep_mask]
        new_edge_type = edge_type[keep_mask]

        # Optionally add noise edges
        if self.add_noise_edges:
            num_noise = int(num_drop * 0.5)
            noise_src = torch.randint(0, N, (num_noise,), device=edge_index.device)
            noise_dst = torch.randint(0, N, (num_noise,), device=edge_index.device)
            noise_edges = torch.stack([noise_src, noise_dst])
            noise_types = torch.zeros(num_noise, dtype=torch.long, device=edge_index.device)

            new_edge_index = torch.cat([new_edge_index, noise_edges], dim=1)
            new_edge_type = torch.cat([new_edge_type, noise_types])

        return node_features, new_edge_index, new_edge_type


class AdversarialAttack(Perturbation):
    """
    Gradient-based adversarial perturbation (FGSM / PGD).
    
    This is the strongest test of model robustness. It computes
    the worst-case input perturbation that maximizes prediction error.
    
    FGSM (Fast Gradient Sign Method):
    x_adv = x + ε * sign(∇_x L(model(x), y))
    
    PGD (Projected Gradient Descent):
    Iterative version of FGSM with projection onto ε-ball.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_steps: int = 5,
        step_size: Optional[float] = None,
        method: str = 'pgd',  # 'fgsm' or 'pgd'
        seed: Optional[int] = None,
    ):
        super().__init__(epsilon, seed)
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size or (epsilon / max(num_steps, 1) * 2.0)
        self.method = method

    def apply(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        model: Optional[nn.Module] = None,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Requires model, targets, and loss_fn for gradient computation.
        Falls back to random noise if model is not provided.
        """
        if model is None or targets is None or loss_fn is None:
            # Fallback: uniform random noise
            noise = torch.empty_like(node_features).uniform_(
                -self.epsilon, self.epsilon
            )
            return node_features + noise, edge_index, edge_type

        # Ensure model is in eval mode for consistent behavior
        was_training = model.training
        model.eval()

        if self.method == 'fgsm':
            adv_features = self._fgsm(
                model, node_features, edge_index, edge_type,
                targets, loss_fn, **kwargs
            )
        else:
            adv_features = self._pgd(
                model, node_features, edge_index, edge_type,
                targets, loss_fn, **kwargs
            )

        if was_training:
            model.train()

        return adv_features, edge_index, edge_type

    def _fgsm(
        self, model, features, edge_index, edge_type,
        targets, loss_fn, **kwargs
    ):
        features = features.clone().detach().requires_grad_(True)

        predictions = model(
            features, edge_index, edge_type,
            kwargs.get('category_ids', {}),
            kwargs.get('dept_ids'),
            kwargs.get('historical_mean'),
        )
        loss = loss_fn(predictions, targets)
        loss.backward()

        # FGSM: perturb in direction of gradient sign
        perturbation = self.epsilon * features.grad.sign()
        adv_features = features.detach() + perturbation

        return adv_features

    def _pgd(
        self, model, features, edge_index, edge_type,
        targets, loss_fn, **kwargs
    ):
        adv_features = features.clone().detach()
        # Random start within ε-ball
        adv_features = adv_features + torch.empty_like(adv_features).uniform_(
            -self.epsilon, self.epsilon
        )

        for _ in range(self.num_steps):
            adv_features = adv_features.clone().detach().requires_grad_(True)

            predictions = model(
                adv_features, edge_index, edge_type,
                kwargs.get('category_ids', {}),
                kwargs.get('dept_ids'),
                kwargs.get('historical_mean'),
            )
            loss = loss_fn(predictions, targets)
            loss.backward()

            # PGD step
            grad = adv_features.grad
            adv_features = adv_features.detach() + self.step_size * grad.sign()

            # Project back to ε-ball around original
            perturbation = adv_features - features
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adv_features = features + perturbation

        return adv_features.detach()
