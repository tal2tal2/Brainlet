import torch


def gating_regularization(gating_weights, lambda_peaky=0.1, lambda_diverse=0.1):
    """
    Regularization term to encourage peaky (low entropy) and diverse (high entropy) gating.
    """
    sample_entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()
    p_mean = gating_weights.mean(dim=0)
    avg_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
    return lambda_peaky * sample_entropy - lambda_diverse * avg_entropy
