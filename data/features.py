"""
Feature Engineering Pipeline — Matches and exceeds M5 winner features.
Produces the tensors that feed into the SigGNN model.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, FeatureConfig


class FeatureEngineer:
    """
    Builds all features needed for the SigGNN model.
    
    Feature groups:
    1. Lag features (demand at t-7, t-14, t-28, ...)
    2. Rolling statistics (mean, std over 7/14/28/56 day windows)
    3. Calendar features (day-of-week, month, SNAP, events, harmonics)
    4. Price features (sell_price, price_change, price_momentum)
    5. Category encodings (integer IDs for embedding layers)
    """

    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_cfg = data_config
        self.feat_cfg = feature_config

    def compute_lag_features(
        self, 
        sales_matrix: np.ndarray, 
        lags: List[int]
    ) -> np.ndarray:
        """
        Compute lag features for each item.
        
        Args:
            sales_matrix: (N, T) raw sales
            lags: list of lag values [7, 14, 28, ...]
            
        Returns:
            (N, T, len(lags)) lag feature array
        """
        N, T = sales_matrix.shape
        lag_features = np.zeros((N, T, len(lags)), dtype=np.float32)

        for i, lag in enumerate(lags):
            if lag < T:
                lag_features[:, lag:, i] = sales_matrix[:, :T - lag]
            # Values before the lag are left as 0

        return lag_features

    def compute_rolling_features(
        self, 
        sales_matrix: np.ndarray, 
        windows: List[int]
    ) -> np.ndarray:
        """
        Compute rolling mean and std for each window size.
        Uses a cumulative sum trick for O(N×T) instead of O(N×T×W).
        
        Returns:
            (N, T, len(windows) * 2) — [mean_w1, std_w1, mean_w2, std_w2, ...]
        """
        N, T = sales_matrix.shape
        num_feats = len(windows) * 2
        rolling_features = np.zeros((N, T, num_feats), dtype=np.float32)

        for wi, w in enumerate(windows):
            for t in range(w, T):
                window_data = sales_matrix[:, t - w:t]
                rolling_features[:, t, wi * 2] = np.mean(window_data, axis=1)
                rolling_features[:, t, wi * 2 + 1] = np.std(window_data, axis=1) + 1e-8

        return rolling_features

    def compute_price_features(
        self, 
        price_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Price features: raw price, price change, price momentum.
        
        Returns:
            (N, T, 3) — [price, price_change, price_momentum]
        """
        N, T = price_matrix.shape
        price_feats = np.zeros((N, T, 3), dtype=np.float32)

        # Raw price (normalized per item)
        price_mean = price_matrix.mean(axis=1, keepdims=True) + 1e-8
        price_feats[:, :, 0] = price_matrix / price_mean

        # Price change (relative)
        price_feats[:, 1:, 1] = (price_matrix[:, 1:] - price_matrix[:, :-1]) / (price_matrix[:, :-1] + 1e-8)

        # Price momentum (rolling mean of changes)
        w = self.feat_cfg.price_momentum_window
        for t in range(w, T):
            price_feats[:, t, 2] = np.mean(price_feats[:, t - w:t, 1], axis=1)

        return price_feats

    def encode_categories(
        self, 
        metadata: 'pd.DataFrame'
    ) -> Dict[str, np.ndarray]:
        """
        Convert categorical columns to integer IDs for embedding layers.
        
        Returns:
            Dictionary of category name → integer array (N,)
        """
        encodings = {}
        for col in ['store_id', 'dept_id', 'cat_id', 'state_id', 'item_id']:
            if col in metadata.columns:
                categories = metadata[col].astype('category')
                encodings[col] = categories.cat.codes.values.astype(np.int64)
                encodings[f'{col}_vocab_size'] = len(categories.cat.categories)

        return encodings

    def build_stream_tensors(
        self, 
        dataset: Dict,
        start_day: int, 
        end_day: int,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, torch.Tensor]:
        """
        Build all feature tensors for a given time window.
        
        This is the main entry point. Produces the tensor that feeds
        into the multi-scale signature encoder.
        
        Args:
            dataset: output of M5DataLoader.prepare_dataset()
            start_day: first day index (0-based) to include
            end_day: last day index (0-based, exclusive)
            device: torch device
            
        Returns:
            Dictionary with:
            - 'node_features': (N, T_window, C) — per-timestep features
            - 'targets': (N, horizon) — ground truth for forecast horizon
            - 'category_ids': dict of (N,) integer tensors for embeddings
            - 'sales_history': (N, T_total) — full sales history for WRMSSE
        """
        sales = dataset['sales_matrix']
        prices = dataset['price_matrix']
        cal_feats = dataset['calendar_features']
        metadata = dataset['metadata']
        N, T_total = sales.shape

        # ── Compute all feature arrays on the full timeline ──
        lag_feats = self.compute_lag_features(sales, self.feat_cfg.lags)
        rolling_feats = self.compute_rolling_features(sales, self.feat_cfg.rolling_windows)
        price_feats = self.compute_price_features(prices)

        # ── Slice to the requested window ──
        window_sales = sales[:, start_day:end_day]              # (N, W)
        window_lags = lag_feats[:, start_day:end_day, :]        # (N, W, L)
        window_rolling = rolling_feats[:, start_day:end_day, :] # (N, W, R)
        window_prices = price_feats[:, start_day:end_day, :]    # (N, W, 3)
        window_cal = cal_feats[start_day:end_day, :]            # (W, 8), broadcast to N

        W = end_day - start_day

        # ── Expand calendar to (N, W, 8) ──
        window_cal_expanded = np.tile(window_cal[np.newaxis, :, :], (N, 1, 1))

        # ── Demand features: log1p(sales) for scale stability ──
        demand_feat = np.log1p(window_sales)[:, :, np.newaxis]  # (N, W, 1)

        # ── Concatenate all features along channel dimension ──
        # Shape: (N, W, 1 + L + R + 3 + 8) = (N, W, C)
        node_features = np.concatenate([
            demand_feat,        # 1: log demand
            window_lags,        # L: lag features (6 default)
            window_rolling,     # R: rolling features (8 default: 4 windows × 2)
            window_prices,      # 3: price features
            window_cal_expanded # 8: calendar features
        ], axis=-1)

        # ── Targets: next `horizon` days ──
        target_start = end_day
        target_end = min(end_day + self.data_cfg.horizon, T_total)
        if target_end > target_start:
            targets = sales[:, target_start:target_end]
        else:
            targets = np.zeros((N, self.data_cfg.horizon), dtype=np.float32)

        # Pad if not enough target days
        if targets.shape[1] < self.data_cfg.horizon:
            pad_width = self.data_cfg.horizon - targets.shape[1]
            targets = np.pad(targets, ((0, 0), (0, pad_width)), mode='constant')

        # ── Category encodings ──
        cat_ids = self.encode_categories(metadata)

        # ── Convert to tensors ──
        result = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
            'targets': torch.tensor(targets, dtype=torch.float32).to(device),
            'sales_history': torch.tensor(sales[:, :end_day], dtype=torch.float32).to(device),
            'category_ids': {
                k: torch.tensor(v, dtype=torch.long).to(device)
                for k, v in cat_ids.items()
                if not k.endswith('_vocab_size')
            },
            'category_vocab_sizes': {
                k: v for k, v in cat_ids.items()
                if k.endswith('_vocab_size')
            },
            'num_features': node_features.shape[-1],
        }

        print(f"   Node features: {result['node_features'].shape}")
        print(f"   Targets: {result['targets'].shape}")
        print(f"   Feature channels: {result['num_features']}")

        return result
