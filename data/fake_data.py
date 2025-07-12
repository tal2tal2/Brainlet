import numpy as np
import torch
from torch.utils.data import Dataset


class RandomSeriesDataset(Dataset):
    """
    Dataset class that creates samples from the generated time series.
    Each sample consists of an input x_t and a target which is the concatenation
    of the derivative f_t and the next state x_{t+1}.
    """

    def __init__(self, series_list, dynamics, dt=0.001):
        self.samples = []
        self.dt = dt
        for series, states in series_list:
            for t in range(len(series) - 1):
                x_t = series[t]
                state_t = states[t]
                x_next = series[t + 1]
                f_t, _ = dynamics(x_t, state_t)
                # Target: [derivative, next state]
                target = np.concatenate([f_t, x_next])
                self.samples.append((x_t, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, target = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
