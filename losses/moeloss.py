import torch

from settings import ModelConfig, GeneratorConfig


class MoELoss(torch.nn.Module):
    def __init__(self, generator: GeneratorConfig, model: ModelConfig):
        super(MoELoss, self).__init__()
        self.lambda_peaky = model.lambda_peaky
        self.lambda_diverse = model.lambda_diverse
        self.lambda_phys = model.lambda_phys
        self.mse = torch.nn.MSELoss()
        self.dt = generator.dt
        self.target_len = generator.target_len

    def __call__(self, outputs, targets, inputs, weights):
        loss_mse = self.mse(outputs, targets)
        f_pred = outputs[:, :, :2]  # First two elements are the predicted derivative
        x_next_pred = outputs[:, :, 2:]  # Last two elements are the predicted next state
        combine_phys = torch.cat([inputs[:, -1:, 0:2], targets[:, :-1, 2:]], dim=1)

        sample_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        p_mean = weights.mean(dim=(0, 1))
        avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
        loss_peaky = self.lambda_peaky * sample_entropy
        loss_diverse = - self.lambda_diverse * avg_entropy

        physics_loss = self.lambda_phys * self.mse(x_next_pred, combine_phys + self.dt * f_pred)
        return loss_mse, physics_loss, loss_peaky, loss_diverse
