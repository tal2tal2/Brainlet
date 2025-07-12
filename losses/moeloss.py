import torch

from models.moe.utils import gating_regularization


class MoELoss(torch.nn.Module):
    def __init__(self, dt, lambda_peaky=0.1, lambda_diverse=0.1, lambda_phys=0.1):
        super(MoELoss, self).__init__()
        self.lambda_peaky = lambda_peaky
        self.lambda_diverse = lambda_diverse
        self.lambda_phys = lambda_phys
        self.mse = torch.nn.MSELoss()
        self.dt = dt

    def __call__(self, outputs, targets, inputs, weights):
        loss_mse = self.mse(outputs, targets)
        loss_reg = gating_regularization(weights, lambda_peaky=0.1, lambda_diverse=0.1)
        f_pred = outputs[:, :2]  # First two elements are the predicted derivative
        x_next_pred = outputs[:, 2:]  # Last two elements are the predicted next state
        physics_loss = self.mse(x_next_pred, inputs + self.dt * f_pred)
        loss = loss_mse + loss_reg + self.lambda_phys * physics_loss
        return loss
