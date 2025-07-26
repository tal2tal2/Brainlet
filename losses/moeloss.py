import torch

from settings import ModelConfig, GeneratorConfig


class MoELoss(torch.nn.Module):
    def __init__(self, generator: GeneratorConfig, model: ModelConfig):
        super(MoELoss, self).__init__()
        self.lambda_peaky = model.lambda_peaky
        self.lambda_diverse = model.lambda_diverse
        self.lambda_phys = model.lambda_phys
        self.lambda_time = model.lambda_time
        self.mse = torch.nn.MSELoss(reduction="none")
        self.derivative = generator.derivative
        self.dt = generator.dt
        self.target_len = generator.target_len
        self.input_len = generator.input_len

    def linear_time_weights(self, device):
        time_weights = torch.linspace(1.0, self.input_len, steps=self.target_len, device=device)
        time_weights = time_weights / time_weights.sum()
        return time_weights

    def get_time_weights(self, device):
        weights = torch.linspace(0, 1, steps=self.target_len, device=device)
        weights = torch.exp(weights * self.lambda_time)
        return weights / weights.sum()

    def __call__(self, outputs, targets, inputs, weights):
        loss_mse = (self.mse(outputs, targets).mean(dim=-1) * self.get_time_weights(targets.device)).mean()
        f_pred = outputs[:, :, :2]  # First two elements are the predicted derivative
        if not self.derivative: f_pred *= self.dt
        x_next_pred = outputs[:, :, 2:]  # Last two elements are the predicted next state
        combine_phys = torch.cat([inputs[:, -1:, 0:2], targets[:, :-1, 2:]], dim=1)

        sample_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        p_mean = weights.mean(dim=(0, 1))
        avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
        loss_peaky = self.lambda_peaky * sample_entropy
        loss_diverse = - self.lambda_diverse * avg_entropy
        physical_mse = (self.mse(x_next_pred, combine_phys + f_pred).mean(dim=-1) * self.get_time_weights(
            targets.device)).mean()

        physics_loss = self.lambda_phys * physical_mse
        return loss_mse, physics_loss, loss_peaky, loss_diverse
