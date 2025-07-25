import numpy as np
import torch
from torch.utils.data import Dataset

from settings import Config, GeneratorConfig
from utils.lds import generate_time_series, continuous_dynamics


class RandomSeriesDataset(Dataset):
    """
    Dataset class that creates samples from the generated time series.
    Each sample consists of an input x_t and a target which is the concatenation
    of the derivative f_t and the next state x_{t+1}.
    """

    def __init__(self, series_list, dynamics, generator: GeneratorConfig):
        self.generator = generator
        self.samples = []
        self.dt = self.generator.dt
        input_len = self.generator.input_len
        target_len = self.generator.target_len
        for series, states in series_list:
            for t in range(len(series)-input_len-target_len+1): # TODO: do i want to have larger steps?
                x_t = series[t:t+input_len]
                x_next = series[t+input_len:t+input_len+target_len]
                fts = []
                state_list=[]
                # need loop to predict next derivatives
                for i in range(target_len):
                    idx = t + input_len + i
                    f_t, _ = dynamics(series[idx], states[idx])
                    fts.append(f_t)
                    state_list.append(states[idx])
                target = np.concatenate([np.stack(fts, axis=0), x_next], axis=1)
                state_list = np.stack(state_list, axis=0)
                self.samples.append((x_t, target, state_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, target, states = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), torch.tensor(states, dtype=torch.int8)

def test():
    config = Config()
    series_list = generate_time_series(**config.generator.get_generator_params())
    dataset = RandomSeriesDataset(series_list, continuous_dynamics, config.generator)
    print([(x.shape, y.shape) for x,y in dataset[:5]])


if __name__ == "__main__":
    test()
