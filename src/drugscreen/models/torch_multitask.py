"""PyTorch shared MLP with masked heterogeneous task losses."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

class MultiTaskNet(nn.Module):
    def __init__(self, n_features, n_tasks, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 128), nn.ReLU())
        self.head = nn.Linear(128, n_tasks)

    def forward(self, x):
        return self.head(self.encoder(x))

@dataclass
class TorchMaskedMultiTask:
    task_names: list[str]
    task_types: dict[str, str]
    epochs: int = 300
    learning_rate: float = 1e-3
    seed: int = 42

    def fit(self, x, y, train_mask, validation_mask=None):
        torch.manual_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        train_mask = np.asarray(train_mask, dtype=bool)
        self.x_mean_ = x.mean(0).astype(np.float32)
        self.x_scale_ = x.std(0).astype(np.float32); self.x_scale_[self.x_scale_ < 1e-6] = 1
        x = (x - self.x_mean_) / self.x_scale_
        self.y_mean_ = np.zeros(y.shape[1], np.float32); self.y_scale_ = np.ones(y.shape[1], np.float32)
        target = np.nan_to_num(y, nan=0.0).astype(np.float32)
        for i, task in enumerate(self.task_names):
            values = y[train_mask[:, i] & ~np.isnan(y[:, i]), i]
            if self.task_types[task] == "regression" and len(values):
                self.y_mean_[i] = values.mean(); self.y_scale_[i] = max(values.std(), 1e-6)
                target[:, i] = (target[:, i] - self.y_mean_[i]) / self.y_scale_[i]
        self.net_ = MultiTaskNet(x.shape[1], y.shape[1])
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        tx, ty, tm = map(torch.tensor, (x, target, train_mask))
        vm = torch.tensor(np.asarray(validation_mask, dtype=bool)) if validation_mask is not None else None
        best_state, best_loss = None, float("inf")
        for _ in range(self.epochs):
            self.net_.train(); optimizer.zero_grad(); output = self.net_(tx); loss = self._loss(output, ty, tm); loss.backward(); optimizer.step()
            if vm is not None:
                self.net_.eval()
                with torch.no_grad(): current = float(self._loss(self.net_(tx), ty, vm))
                if current < best_loss:
                    best_loss, best_state = current, {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
        if best_state is not None: self.net_.load_state_dict(best_state)
        return self

    def _loss(self, output, target, mask):
        losses = []
        for i, task in enumerate(self.task_names):
            observed = mask[:, i]
            if not torch.any(observed): continue
            if self.task_types[task] == "classification":
                value = nn.functional.binary_cross_entropy_with_logits(output[observed, i], target[observed, i])
            else:
                value = nn.functional.mse_loss(output[observed, i], target[observed, i])
            losses.append(value)
        return torch.stack(losses).mean()

    def predict(self, x):
        x = torch.tensor((np.asarray(x, dtype=np.float32) - self.x_mean_) / self.x_scale_)
        self.net_.eval()
        with torch.no_grad(): output = self.net_(x).numpy()
        result = output.copy()
        for i, task in enumerate(self.task_names):
            if self.task_types[task] == "classification": result[:, i] = 1 / (1 + np.exp(-np.clip(output[:, i], -30, 30)))
            else: result[:, i] = output[:, i] * self.y_scale_[i] + self.y_mean_[i]
        return result
