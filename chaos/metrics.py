"""
Resilience Metrics for Chaos Engineering Evaluation.
"""
import numpy as np
from typing import Dict, List


class ResilienceMetrics:
    """Computes principled resilience metrics from chaos experiment results."""

    @staticmethod
    def prediction_stability(results: List[Dict]) -> float:
        """Average stability across all perturbation types."""
        scores = [r['stability_score'] for r in results]
        return float(np.mean(scores))

    @staticmethod
    def worst_case_stability(results: List[Dict]) -> float:
        """Minimum stability (worst perturbation)."""
        scores = [r['stability_score'] for r in results]
        return float(np.min(scores))

    @staticmethod
    def robustness_profile(results: List[Dict]) -> Dict[str, float]:
        """Categorized robustness scores."""
        categories = {
            'demand': ['demand_shock_spike', 'demand_shock_crash'],
            'supply': ['supply_disruption'],
            'economic': ['price_volatility'],
            'temporal': ['calendar_shift'],
            'structural': ['graph_corruption_10', 'graph_corruption_30'],
            'adversarial': ['adversarial_fgsm', 'adversarial_pgd'],
        }
        profile = {}
        for cat, names in categories.items():
            cat_results = [r for r in results if r['perturbation'] in names]
            if cat_results:
                profile[cat] = float(np.mean([r['stability_score'] for r in cat_results]))
        
        profile['overall'] = float(np.mean(list(profile.values()))) if profile else 0.0
        return profile

    @staticmethod
    def summary_table(results: List[Dict]) -> str:
        """Pretty-print results table."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'Perturbation':<30} {'Stability':>10} {'Mean Δ':>12} {'Rel Δ':>10}")
        lines.append("=" * 70)
        for r in results:
            lines.append(
                f"{r['perturbation']:<30} "
                f"{r['stability_score']:>10.4f} "
                f"{r['mean_deviation']:>12.4f} "
                f"{r['relative_change']:>10.4f}"
            )
        lines.append("=" * 70)
        
        profile = ResilienceMetrics.robustness_profile(results)
        lines.append(f"\nOverall Resilience Score: {profile.get('overall', 0):.4f}")
        for cat, score in profile.items():
            if cat != 'overall':
                lines.append(f"  {cat:<20}: {score:.4f}")
        
        return "\n".join(lines)
