"""
Training loop for SigGNN with Tweedie loss, cosine scheduling,
mixed precision, and optional adversarial training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Optional, Tuple


class Trainer:
    """Full training pipeline for SigGNN."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        patience: int = 15,
        gradient_clip: float = 1.0,
        warmup_epochs: int = 5,
        use_amp: bool = False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=lr * 0.01
        )

        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_no_improve = 0

    def _warmup_lr(self, epoch: int):
        """Linear warmup for first N epochs."""
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * factor

    def _forward_model(self, node_features, edge_index, edge_type,
                       category_ids, dept_ids, historical_mean):
        """Run model forward pass."""
        return self.model(
            node_features, edge_index, edge_type,
            category_ids, dept_ids, historical_mean,
        )

    def train_epoch(
        self,
        node_features, edge_index, edge_type, targets,
        category_ids, dept_ids=None, historical_mean=None,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                predictions = self._forward_model(
                    node_features, edge_index, edge_type,
                    category_ids, dept_ids, historical_mean,
                )
                loss = self.loss_fn(predictions, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self._forward_model(
                node_features, edge_index, edge_type,
                category_ids, dept_ids, historical_mean,
            )
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

        return loss.item()



    @torch.no_grad()
    def validate(
        self,
        node_features, edge_index, edge_type, targets,
        category_ids, dept_ids=None, historical_mean=None,
    ) -> Tuple[float, torch.Tensor]:
        """Validate and return loss + predictions."""
        self.model.eval()
        predictions = self._forward_model(
            node_features, edge_index, edge_type,
            category_ids, dept_ids, historical_mean,
        )
        loss = self.loss_fn(predictions, targets)
        return loss.item(), predictions

    def train(
        self,
        # Training data
        train_features, train_edge_index, train_edge_type, train_targets,
        train_category_ids, train_dept_ids=None, train_historical_mean=None,
        # Validation data (separate!)
        val_features=None, val_edge_index=None, val_edge_type=None,
        val_targets=None, val_category_ids=None, val_dept_ids=None,
        val_historical_mean=None,
    ) -> Dict:
        """Full training loop with early stopping."""
        print(f"\n🚀 Training SigGNN | {self.model.count_parameters():,} parameters")
        print(f"   Epochs: {self.max_epochs} | Patience: {self.patience} | AMP: {self.use_amp}")

        train_category_ids = train_category_ids if train_category_ids is not None else {}
        has_val = val_features is not None and val_targets is not None
        
        if has_val:
            val_category_ids = val_category_ids if val_category_ids is not None else {}
        
        t0 = time.time()

        for epoch in range(self.max_epochs):
            self._warmup_lr(epoch)

            # ── Train ──
            train_loss = self.train_epoch(
                train_features, train_edge_index, train_edge_type,
                train_targets, train_category_ids, train_dept_ids,
                train_historical_mean,
            )
            self.train_losses.append(train_loss)

            # ── LR schedule ──
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # ── Validate ──
            if has_val:
                v_ei = val_edge_index if val_edge_index is not None else train_edge_index
                v_et = val_edge_type if val_edge_type is not None else train_edge_type

                val_loss, _ = self.validate(
                    val_features, v_ei, v_et, val_targets,
                    val_category_ids, val_dept_ids, val_historical_mean,
                )
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss - 1e-5:
                    self.best_val_loss = val_loss
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    self.epochs_no_improve = 0
                    marker = " ✓ best"
                else:
                    self.epochs_no_improve += 1
                    marker = ""

                if epoch % 5 == 0 or marker:
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - t0
                    print(f"   Epoch {epoch:03d} | Train: {train_loss:.4f} | "
                          f"Val: {val_loss:.4f} | LR: {lr:.6f} | "
                          f"Time: {elapsed:.1f}s{marker}")

                if self.epochs_no_improve >= self.patience:
                    print(f"\n   ⏹ Early stopping at epoch {epoch} "
                          f"(best val: {self.best_val_loss:.4f})")
                    break
            else:
                if epoch % 5 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - t0
                    print(f"   Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                          f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"   Loaded best model (val_loss={self.best_val_loss:.4f})")

        total_time = time.time() - t0
        print(f"\n✅ Training complete in {total_time:.1f}s")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'epochs_trained': len(self.train_losses),
        }
