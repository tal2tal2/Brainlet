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
from utils.lds import match_expert_to_state


class MixtureOfExperts(LightningModule):
    def __init__(self, config: ModelConfig, generator: GeneratorConfig, input_dim=2, output_dim=4):
        super(MixtureOfExperts, self).__init__()
        self.config = config
        self.generator = generator
        if self.generator.derivative:
            input_dim = input_dim * 3
        if self.generator.time:
            input_dim = input_dim + 1
        # self.domains = nn.ModuleList([
        #     nn.Identity(),  # raw time‐series
        #     lambda x: x[:, 1:, :] - x[:, :-1, :],  # velocity (B,L_in-1,D) – you may pad/upsample
        #     lambda x: torch.fft.rfft(x, dim=1).abs(),  # magnitude spectrum
        #     lambda x: torch.stack([torch.sin(torch.fft.rfft(x, dim=1).angle()),
        #                     torch.cos(torch.fft.rfft(x, dim=1).angle())], dim=-1), # phase fft
        # ])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_dim * self.generator.input_len, output_dim * self.generator.target_len),
                # nn.Tanh(),
                # nn.Linear(128, output_dim * self.generator.target_len),
                nn.Unflatten(1, (self.generator.target_len, output_dim)),
            ) for _ in range(self.config.num_experts)
        ])
        self.gating_network = nn.Sequential(  # how much should we believe each expert
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim * self.generator.input_len, self.config.num_experts * self.generator.target_len),
            # nn.Tanh(),
            # nn.Linear(128, self.config.num_experts * self.generator.target_len),
            nn.Unflatten(1, (self.generator.target_len, self.config.num_experts)),
            nn.Softmax(dim=-1)
        )
        self.augmentations = v2.Identity()
        self.criterion = MoELoss(self.generator, self.config)

        self.test_gating = []

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.val_preds = []
        self.val_targets = []
        self.epoch = 0
        self.accuracy = []

    def forward(self, x):
        gating_weights = self.gating_network(x)  # Shape: [batch, num_experts] TODO:maybe change for long inputs?
        # Stack expert outputs: shape becomes [batch, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts' outputs weighted by the gating network
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=-1)
        return final_output, gating_weights

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        inputs, targets, states = batch
        inputs = self.augmentations(inputs)
        losses, _ = self._run_batch([inputs, targets, states], calculate_metrics=False)
        loss_mse, physics_loss, loss_peaky, loss_diverse = losses
        loss = loss_mse + physics_loss + loss_peaky + loss_diverse
        self.log('train_loss_mse', loss_mse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('train_loss_physics', physics_loss, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('train_loss_peaky', loss_peaky, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('train_loss_diverse', loss_diverse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True, on_step=False)
        return loss

    def on_train_epoch_end(self) -> None:
        self.epoch += 1

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        losses, metrics = self._run_batch(batch)
        loss_mse, physics_loss, loss_peaky, loss_diverse = losses
        loss = loss_mse + physics_loss + loss_peaky + loss_diverse
        self.log('val_loss_mse', loss_mse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('val_loss_physics', physics_loss, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('val_loss_peaky', loss_peaky, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('val_loss_diverse', loss_diverse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False)
        val_metrics = {f'val_{key}': value for key, value in metrics.items()}
        self.log_dict(val_metrics, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.on_metric_epoch_end('val_')

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        losses, metrics = self._run_batch(batch, test=True)
        loss_mse, physics_loss, loss_peaky, loss_diverse = losses
        loss = loss_mse + physics_loss + loss_peaky + loss_diverse
        self.log('test_loss_mse', loss_mse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('test_loss_physics', physics_loss, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('test_loss_peaky', loss_peaky, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('test_loss_diverse', loss_diverse, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        self.log('test_loss', loss, sync_dist=True, on_step=False)
        test_metrics = {f'test_{key}': value for key, value in metrics.items()}
        self.log_dict(test_metrics, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        return loss

    def on_test_epoch_end(self) -> None:
        self.on_metric_epoch_end('test_')
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
        inputs, targets, _ = batch
        preds, weights = self(inputs)
        deriv_pred = preds[:, :, :2]
        direct_pred = preds[:, :, 2:]
        x_0 = inputs[:, -1:, :2]
        delta = deriv_pred * self.generator.dt
        deriv_series = x_0 + delta.cumsum(dim=1)
        state_series = weights.argmax(dim=-1)

        return deriv_series, direct_pred, state_series

    def _run_batch(self, batch: list[Tensor], calculate_metrics: bool = True, test: bool = False) -> tuple[
        Tensor, dict[str, Tensor]]:

        inputs, targets, states = batch
        outputs, gating_weights = self(inputs)
        metrics = self._calculate_metrics(outputs.clone(), targets.clone(), gating_weights,
                                          states) if calculate_metrics else {}
        if test: self._calculate_test_metrics(gating_weights)
        losses = self.criterion(outputs, targets, inputs, gating_weights)
        return losses, metrics

    def _calculate_test_metrics(self, gating_weights):
        self.test_gating.append(gating_weights.detach())

    def _calculate_metrics(self, outputs, targets, gating_weights, states):
        outputs_flat = torch.flatten(outputs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)
        metrics = {
            "MSE": self.mse(outputs_flat, targets_flat),
            "MAE": self.mae(outputs_flat, targets_flat),
            "R2": self.r2(outputs_flat, targets_flat),
            "gating_entropy": self.gating_entropy(gating_weights),
            "gating_sparsity": (gating_weights > 0.1).float().sum(dim=1).mean(),
            "MSE_t_0": self.mse(outputs[:, 0, :].reshape(-1), targets[:, 0, :].reshape(-1)),
            "MSE_t_end": self.mse(outputs[:, -1, :].reshape(-1), targets[:, -1, :].reshape(-1)),
            "R2_t_0": self.r2(outputs[:, 0, :].reshape(-1), targets[:, 0, :].reshape(-1)),
            "R2_t_end": self.r2(outputs[:, -1, :].reshape(-1), targets[:, -1, :].reshape(-1)),
        }
        top_expert = torch.argmax(gating_weights, dim=1)  # shape: [batch_size]
        for i in range(self.config.num_experts):
            freq = (top_expert == i).float().mean()
            metrics.update({f"expert_{i}_usage": freq})
        self.val_preds.append(gating_weights.argmax(dim=-1))
        self.val_targets.append(states)
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

    def on_metric_epoch_end(self, name) -> None:
        preds = torch.cat(self.val_preds, dim=0).cpu().numpy()  # → (N, L)
        trues = torch.cat(self.val_targets, dim=0).cpu().numpy()  # → (N, L)

        remapped_pred_t_0 = match_expert_to_state(preds[:, 0], trues[:, 0], num_classes=self.config.num_experts)
        remapped_t_0 = torch.from_numpy(remapped_pred_t_0).cpu()  # (N, L)
        remapped_pred_t_end = match_expert_to_state(preds[:, -1], trues[:, -1], num_classes=self.config.num_experts)
        remapped_t_end = torch.from_numpy(remapped_pred_t_end).cpu()

        # 4) compute accuracies in torch
        acc_t0 = (remapped_t_0 == trues[:, 0]).float().mean()
        acc_tend = (remapped_t_end == trues[:, -1]).float().mean()

        # 5) log
        self.log(f"{name}accuracy_t_0", acc_t0, prog_bar=True, sync_dist=True)
        self.log(f"{name}accuracy_t_end", acc_tend, prog_bar=True, sync_dist=True)

        if name == 'val_':
            self.accuracy.append(acc_t0.detach().item())
            if len(self.accuracy) > 5:
                self.accuracy = self.accuracy[-5:]
            if self.epoch < 5:
                avg_acc = 0.4 + 0.01 * self.epoch
            elif self.epoch < 10:
                weight = (self.epoch - 5) / 5.0
                real_avg = sum(self.accuracy) / len(self.accuracy)
                avg_acc = (1 - weight) * 0.45 + weight * real_avg
            else:
                avg_acc = sum(self.accuracy) / len(self.accuracy)
            self.log(f'{name}avg_accuracy', avg_acc, prog_bar=True, sync_dist=True)

        # 6) clear for next epoch
        self.val_preds.clear()
        self.val_targets.clear()
