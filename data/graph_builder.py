"""
Hierarchical Graph Construction for the M5 Supply Chain.

Constructs a heterogeneous graph with three edge types:
1. Hierarchical edges: item → dept → cat → store → state (known structure)
2. Correlation edges: top-k correlated items within same store
3. Cross-store edges: same item across different stores

Uses COO (edge_index) format for memory-efficient sparse operations.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Edge type constants
EDGE_HIERARCHICAL = 0
EDGE_CORRELATION = 1
EDGE_CROSS_STORE = 2


class HierarchicalGraphBuilder:
    """
    Builds the supply chain graph from M5 metadata and sales data.
    
    The graph operates at the item-store level (bottom of M5 hierarchy).
    Each node represents one (item_id, store_id) combination.
    """

    def __init__(self, top_k_corr: int = 10, corr_threshold: float = 0.5):
        """
        Args:
            top_k_corr: Number of top correlated items to connect per node
            corr_threshold: Minimum correlation to include an edge
        """
        self.top_k_corr = top_k_corr
        self.corr_threshold = corr_threshold

    def build_hierarchical_edges(
        self, 
        metadata: 'pd.DataFrame'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges based on the M5 hierarchy:
        item-store → department, department → category, within-store items.
        
        Two items in the same department at the same store get connected.
        """
        N = len(metadata)
        src_list, dst_list = [], []

        # Group by (store_id, dept_id) — items in same department at same store
        groups = metadata.groupby(['store_id', 'dept_id']).groups
        for group_name, indices in groups.items():
            idx_list = indices.tolist()
            # Connect all pairs within the group (small groups, so O(k²) is fine)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    src_list.extend([idx_list[i], idx_list[j]])
                    dst_list.extend([idx_list[j], idx_list[i]])

        # Also connect items in same category across departments (same store)
        groups_cat = metadata.groupby(['store_id', 'cat_id']).groups
        for group_name, indices in groups_cat.items():
            # Sample connections to avoid explosion (max 5 per item)
            idx_list = indices.tolist()
            if len(idx_list) > 50:
                # Subsample for large groups
                np.random.seed(42)
                sampled = np.random.choice(idx_list, size=min(50, len(idx_list)), replace=False)
                for i in range(len(sampled)):
                    for j in range(i + 1, min(i + 5, len(sampled))):
                        src_list.extend([sampled[i], sampled[j]])
                        dst_list.extend([sampled[j], sampled[i]])

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)
        return src, dst

    def build_correlation_edges(
        self, 
        sales_matrix: np.ndarray,
        metadata: 'pd.DataFrame',
        train_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges based on sales correlation.
        Only uses TRAINING data to avoid leakage.
        
        Connects each item to its top-k most correlated items
        within the same store (to keep graph meaningful).
        """
        N = len(metadata)
        train_sales = sales_matrix[:, :train_end]
        src_list, dst_list = [], []

        # Compute correlation per store to keep it manageable
        store_groups = metadata.groupby('store_id').groups
        for store_id, indices in store_groups.items():
            idx_list = indices.tolist()
            store_sales = train_sales[idx_list]

            # Remove items with zero variance
            stds = store_sales.std(axis=1)
            valid_mask = stds > 1e-6
            valid_idx = [idx_list[i] for i in range(len(idx_list)) if valid_mask[i]]
            valid_sales = store_sales[valid_mask]

            if len(valid_idx) < 2:
                continue

            # Compute correlation matrix
            corr = np.corrcoef(valid_sales)
            np.fill_diagonal(corr, 0)  # No self-loops from correlation

            # Top-k connections per node
            for i in range(len(valid_idx)):
                scores = corr[i]
                # Filter by threshold
                candidates = np.where(scores > self.corr_threshold)[0]
                if len(candidates) == 0:
                    continue

                # Take top-k
                if len(candidates) > self.top_k_corr:
                    top_indices = np.argsort(scores[candidates])[-self.top_k_corr:]
                    candidates = candidates[top_indices]

                for j in candidates:
                    src_list.extend([valid_idx[i], valid_idx[j]])
                    dst_list.extend([valid_idx[j], valid_idx[i]])

            print(f"   Correlation edges for {store_id}: {len(src_list)} edges so far")

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)
        return src, dst

    def build_cross_store_edges(
        self, 
        metadata: 'pd.DataFrame'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Connect the same item across different stores.
        This enables cross-store learning (key M5 winner insight).
        """
        src_list, dst_list = [], []

        item_groups = metadata.groupby('item_id').groups
        for item_id, indices in item_groups.items():
            idx_list = indices.tolist()
            # Connect all stores that carry this item
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    src_list.extend([idx_list[i], idx_list[j]])
                    dst_list.extend([idx_list[j], idx_list[i]])

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)
        return src, dst

    def build_graph(
        self,
        sales_matrix: np.ndarray,
        metadata: 'pd.DataFrame',
        train_end: int,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, torch.Tensor]:
        """
        Build the complete heterogeneous graph.
        
        Returns:
            edge_index: (2, E) tensor of edge indices
            edge_type: (E,) tensor of edge types
            num_nodes: total number of nodes
        """
        print("🔗 Building hierarchical graph...")
        N = len(metadata)

        # ── 1. Hierarchical edges ──
        h_src, h_dst = self.build_hierarchical_edges(metadata)
        h_types = np.full(len(h_src), EDGE_HIERARCHICAL, dtype=np.int64)
        print(f"   Hierarchical edges: {len(h_src)}")

        # ── 2. Correlation edges (train data only!) ──
        c_src, c_dst = self.build_correlation_edges(sales_matrix, metadata, train_end)
        c_types = np.full(len(c_src), EDGE_CORRELATION, dtype=np.int64)
        print(f"   Correlation edges: {len(c_src)}")

        # ── 3. Cross-store edges ──
        x_src, x_dst = self.build_cross_store_edges(metadata)
        x_types = np.full(len(x_src), EDGE_CROSS_STORE, dtype=np.int64)
        print(f"   Cross-store edges: {len(x_src)}")

        # ── Combine all edges ──
        all_src = np.concatenate([h_src, c_src, x_src])
        all_dst = np.concatenate([h_dst, c_dst, x_dst])
        all_types = np.concatenate([h_types, c_types, x_types])

        # ── Add self-loops ──
        self_src = np.arange(N, dtype=np.int64)
        self_dst = np.arange(N, dtype=np.int64)
        self_types = np.full(N, EDGE_HIERARCHICAL, dtype=np.int64)

        all_src = np.concatenate([all_src, self_src])
        all_dst = np.concatenate([all_dst, self_dst])
        all_types = np.concatenate([all_types, self_types])

        # ── Remove duplicate edges ──
        edge_set = set()
        unique_src, unique_dst, unique_types = [], [], []
        for s, d, t in zip(all_src, all_dst, all_types):
            key = (int(s), int(d))
            if key not in edge_set:
                edge_set.add(key)
                unique_src.append(s)
                unique_dst.append(d)
                unique_types.append(t)

        edge_index = torch.tensor(
            np.stack([unique_src, unique_dst]), dtype=torch.long
        ).to(device)
        edge_type = torch.tensor(unique_types, dtype=torch.long).to(device)

        print(f"   Total unique edges: {edge_index.shape[1]}")
        print(f"   Avg degree: {edge_index.shape[1] / N:.1f}")

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_nodes': N,
        }
