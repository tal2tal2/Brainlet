import torch
from torch.utils.data import Dataset


class NeuralTimeSeriesDataset(Dataset):
    """
    Dataset class for neural time series forecasting.
    Each sample consists of a window of neural activity and a target sequence.
    """

    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        target = self.y[idx]

        if self.transform:
            x = self.transform(x)
            target = self.transform(target)

        return x, target
