"""Quick test: can we do a single training step?"""
import torch
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.siggnn import SigGNN, TweedieLoss

vocab_sizes = {
    'store_id_vocab_size': 3, 'dept_id_vocab_size': 5,
    'cat_id_vocab_size': 3, 'state_id_vocab_size': 1,
    'item_id_vocab_size': 50,
}

model = SigGNN(
    input_channels=19, vocab_sizes=vocab_sizes,
    sig_windows=[7, 28, 60], sig_depth=2, use_lead_lag=True,
    gat_hidden=48, gat_heads=2, gat_layers=2,
    gat_edge_types=3, predictor_hidden=96,
    predictor_layers=3, horizon=28, dropout=0.2, num_dept_groups=5,
)

N = 10
node_features = torch.randn(N, 62, 19)
edge_index = torch.tensor([
    list(range(9)) + list(range(1,10)) + list(range(N)),
    list(range(1,10)) + list(range(9)) + list(range(N)),
])
edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
cat_ids = {
    'store_id': torch.randint(0, 3, (N,)),
    'dept_id': torch.randint(0, 5, (N,)),
    'cat_id': torch.randint(0, 3, (N,)),
    'state_id': torch.zeros(N, dtype=torch.long),
    'item_id': torch.randint(0, 50, (N,)),
}
dept_ids = torch.randint(0, 5, (N,))
hist_mean = torch.randn(N).abs() + 1
targets = torch.randn(N, 28).abs()

loss_fn = TweedieLoss(p=1.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

try:
    print("Forward pass...")
    model.train()
    preds = model(node_features, edge_index, edge_type, cat_ids, dept_ids, hist_mean)
    print(f"  Predictions: {preds.shape}")
    print(f"  Pred range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")
    
    print("Computing loss...")
    loss = loss_fn(preds, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    print("Backward pass...")
    loss.backward()
    print("  ✅ Backward PASS")
    
    print("Optimizer step...")
    optimizer.step()
    print("  ✅ Step PASS")
    
except Exception as e:
    traceback.print_exc()
