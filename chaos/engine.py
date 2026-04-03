"""
Chaos Engineering Engine — Orchestrates perturbation experiments.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .perturbations import (
    Perturbation, DemandShock, SupplyDisruption, PriceVolatility,
    CalendarShift, GraphCorruption, AdversarialAttack,
)


class ChaosEngine:
    """
    Orchestrates chaos engineering experiments.
    
    Usage:
        engine = ChaosEngine(config)
        results = engine.run_all(model, features, edge_index, edge_type, targets, ...)
    """

    def __init__(self, num_trials: int = 5, seed: int = 42):
        self.num_trials = num_trials
        self.seed = seed
        self.perturbations = self._build_perturbations()

    def _build_perturbations(self) -> Dict[str, Perturbation]:
        return {
            'demand_shock_spike': DemandShock(severity=0.5, shock_type='spike', seed=self.seed),
            'demand_shock_crash': DemandShock(severity=0.5, shock_type='crash', seed=self.seed+1),
            'supply_disruption': SupplyDisruption(severity=0.5, seed=self.seed+2),
            'price_volatility': PriceVolatility(severity=0.3, seed=self.seed+3),
            'calendar_shift': CalendarShift(severity=0.5, max_shift=3, seed=self.seed+4),
            'graph_corruption_10': GraphCorruption(severity=0.5, drop_ratio=0.1, seed=self.seed+5),
            'graph_corruption_30': GraphCorruption(severity=1.0, drop_ratio=0.3, seed=self.seed+6),
            'adversarial_fgsm': AdversarialAttack(epsilon=0.01, method='fgsm', seed=self.seed+7),
            'adversarial_pgd': AdversarialAttack(epsilon=0.01, num_steps=5, method='pgd', seed=self.seed+8),
        }

    def run_single(
        self,
        name: str,
        perturbation: Perturbation,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        clean_predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        **model_kwargs,
    ) -> Dict[str, float]:
        """Run a single perturbation and measure impact."""
        model.eval()

        with torch.no_grad():
            if isinstance(perturbation, AdversarialAttack) and targets is not None:
                # Adversarial needs gradients
                pass
            
            perturbed_features, perturbed_edges, perturbed_types = perturbation.apply(
                node_features, edge_index, edge_type,
                model=model if isinstance(perturbation, AdversarialAttack) else None,
                targets=targets,
                loss_fn=loss_fn,
                **model_kwargs,
            )

        with torch.no_grad():
            chaos_predictions = model(
                perturbed_features, perturbed_edges, perturbed_types,
                model_kwargs.get('category_ids', {}),
                model_kwargs.get('dept_ids'),
                model_kwargs.get('historical_mean'),
            )

        # Compute stability metrics
        pred_diff = torch.abs(clean_predictions - chaos_predictions)
        mean_deviation = pred_diff.mean().item()
        max_deviation = pred_diff.max().item()
        relative_change = (pred_diff / (torch.abs(clean_predictions) + 1e-8)).mean().item()

        stability = 1.0 - min(relative_change, 1.0)

        return {
            'perturbation': name,
            'mean_deviation': mean_deviation,
            'max_deviation': max_deviation,
            'relative_change': relative_change,
            'stability_score': stability,
        }

    def run_all(
        self,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        **model_kwargs,
    ) -> List[Dict[str, float]]:
        """Run all perturbations and collect results."""
        model.eval()

        with torch.no_grad():
            clean_predictions = model(
                node_features, edge_index, edge_type,
                model_kwargs.get('category_ids', {}),
                model_kwargs.get('dept_ids'),
                model_kwargs.get('historical_mean'),
            )

        results = []
        for name, perturbation in self.perturbations.items():
            print(f"   🔥 Running chaos: {name}...")
            result = self.run_single(
                name, perturbation, model,
                node_features, edge_index, edge_type,
                clean_predictions, targets, loss_fn,
                **model_kwargs,
            )
            results.append(result)
            print(f"      Stability: {result['stability_score']:.4f} | "
                  f"Mean Δ: {result['mean_deviation']:.4f}")

        return results
