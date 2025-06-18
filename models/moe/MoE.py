from typing import Union

import torch
from lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam, AdamW
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from torchvision.transforms import v2

from losses.moe import MoELoss


class MixtureOfExperts(LightningModule):
    def __init__(self, input_dim=2, output_dim=4, num_experts=3):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim)
                # nn.Tanh(),
                # nn.Linear(10, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(  # how much should we believe each expert
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.dt = 0.001
        self.augmentations = v2.Identity()
        self.criterion = MoELoss(self.dt)
        self.model_optimizer = "adam"
        self.lr = 0.01
        self.weight_decay = 0.0001

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()

    def forward(self, x):
        gating_weights = self.gating_network(x)  # Shape: [batch, num_experts]
        # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts' outputs weighted by the gating network
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(1), dim=-1)
        return final_output, gating_weights

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        inputs = self.augmentations(inputs)
        loss, _ = self._run_batch([inputs, targets], calculate_metrics=False)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._run_batch(batch)
        self.log('val_loss', loss, sync_dist=True)
        val_metrics = {f'val_{key}': value for key, value in metrics.items()}
        self.log_dict(val_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._run_batch(batch)
        self.log('test_loss', loss, sync_dist=True)
        test_metrics = {f'test_{key}': value for key, value in metrics.items()}
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _run_batch(self, batch: list[Tensor], calculate_metrics: bool = True) -> tuple[
        Tensor, dict[str, Tensor]]:

        inputs, targets = batch
        outputs, gating_weights = self(inputs)
        metrics = self._calculate_metrics(outputs.clone(), targets.clone(), gating_weights) if calculate_metrics else {}
        loss = self.criterion(outputs, targets, inputs, gating_weights)

        return loss, metrics

    def _calculate_metrics(self, outputs, targets, gating_weights):

        metrics = {
            "MSE": self.mse(outputs, targets),
            "MAE": self.mae(outputs, targets),
            "R2": self.r2(outputs, targets),
            "gating entropy": self.gating_entropy(gating_weights),
            "gating sparsity": (gating_weights > 0.1).float().sum(dim=1).mean()
        }
        top_expert = torch.argmax(gating_weights, dim=1)  # shape: [batch_size]
        for i in range(self.num_experts):
            freq = (top_expert == i).float().mean()
            metrics.update({f"expert_{i}_usage": freq})
        return metrics

    def gating_entropy(self, gating_weights):
        return -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=1).mean()

    def configure_optimizers(self) -> Union[Adam, AdamW]:
        if self.model_optimizer == "adam":
            return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.model_optimizer == "adamw":
            return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.model_optimizer}")
