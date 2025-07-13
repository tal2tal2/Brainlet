from typing import Union

import seaborn as sns
import torch
import wandb
from lightning import LightningModule
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim import Adam, AdamW
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from torchvision.transforms import v2

from losses.moeloss import MoELoss
from settings import ModelConfig, GeneratorConfig


class MixtureOfExperts(LightningModule):
    def __init__(self, config: ModelConfig, generator: GeneratorConfig, input_dim=2, output_dim=4):
        super(MixtureOfExperts, self).__init__()
        self.config = config
        self.generator = generator

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_dim * self.generator.input_len, 128),
                nn.Tanh(),
                nn.Linear(128, output_dim * self.generator.target_len),
                nn.Unflatten(1, (self.generator.target_len, output_dim)),
            ) for _ in range(self.config.num_experts)
        ])
        self.gating_network = nn.Sequential(  # how much should we believe each expert
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim * self.generator.input_len, 128),
            nn.Tanh(),
            nn.Linear(128, self.config.num_experts * self.generator.target_len),
            nn.Unflatten(1, (self.generator.target_len, self.config.num_experts)),
            nn.Softmax(dim=-1)
        )
        self.augmentations = v2.Identity()
        self.criterion = MoELoss(self.generator.dt, self.generator.target_len)

        self.test_gating = []

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()

    def forward(self, x):
        gating_weights = self.gating_network(x)  # Shape: [batch, num_experts] TODO:maybe change for long inputs?
        # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts' outputs weighted by the gating network
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=-1)
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
        loss, metrics = self._run_batch(batch, test=True)
        self.log('test_loss', loss, sync_dist=True)
        test_metrics = {f'test_{key}': value for key, value in metrics.items()}
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self) -> None:
        all_gates = torch.cat(self.test_gating, dim=0).cpu()
        self.test_gating.clear()
        fig = plt.figure(figsize=(8, 6))
        for i in range(all_gates.shape[1]):
            sns.kdeplot(all_gates[:, i], label=f'Expert {i}')
        plt.xlabel('Gating Weight')
        plt.ylabel('Density')
        plt.title('Test Gating Weight Distribution')
        plt.xlim(0, 1)
        sns.despine()
        plt.tight_layout()
        self.logger.experiment.log({
            'test_gating_distribution': wandb.Image(fig)
        })
        plt.close(fig)

    def predict_step(self, batch: list[Tensor], batch_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        inputs, targets = batch
        preds, weights = self(inputs)
        deriv_pred = preds[:,:,:2]
        direct_pred = preds[:,:,2:]
        x_0 = inputs[:,-1:,:]
        delta = deriv_pred*self.generator.dt
        deriv_series = x_0 + delta.cumsum(dim=1)
        state_series = weights.argmax(dim=-1)

        return deriv_series, direct_pred, state_series

    def _run_batch(self, batch: list[Tensor], calculate_metrics: bool = True, test: bool = False) -> tuple[
        Tensor, dict[str, Tensor]]:

        inputs, targets = batch
        outputs, gating_weights = self(inputs)
        metrics = self._calculate_metrics(outputs.clone(), targets.clone(), gating_weights) if calculate_metrics else {}
        if test: self._calculate_test_metrics(gating_weights)
        loss = self.criterion(outputs, targets, inputs, gating_weights)

        return loss, metrics

    def _calculate_test_metrics(self, gating_weights):
        self.test_gating.append(gating_weights.detach())

    def _calculate_metrics(self, outputs, targets, gating_weights):
        outputs_flat = torch.flatten(outputs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)
        metrics = {
            "MSE": self.mse(outputs_flat, targets_flat),
            "MAE": self.mae(outputs_flat, targets_flat),
            "R2": self.r2(outputs_flat, targets_flat),
            "gating entropy": self.gating_entropy(gating_weights),
            "gating sparsity": (gating_weights > 0.1).float().sum(dim=1).mean(),
            "MSE_t_0": self.mse(outputs[:,0,:].reshape(-1),targets[:,0,:].reshape(-1)),
            "MSE_t_end": self.mse(outputs[:,-1,:].reshape(-1),targets[:,-1,:].reshape(-1)),
            "R2_t_0": self.r2(outputs[:, 0, :].reshape(-1), targets[:, 0, :].reshape(-1)),
            "R2_t_end": self.r2(outputs[:, -1, :].reshape(-1), targets[:, -1, :].reshape(-1)),
        }
        top_expert = torch.argmax(gating_weights, dim=1)  # shape: [batch_size]
        for i in range(self.config.num_experts):
            freq = (top_expert == i).float().mean()
            metrics.update({f"expert_{i}_usage": freq})
        return metrics

    def gating_entropy(self, gating_weights):
        return -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()

    def configure_optimizers(self) -> Union[Adam, AdamW]:
        if self.config.model_optimizer == "adam":
            return Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.model_optimizer == "adamw":
            return AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.model_optimizer}")
