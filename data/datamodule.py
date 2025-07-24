import os

from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from data.fake_data import RandomSeriesDataset
from utils.lds import generate_time_series, continuous_dynamics, load_subset


class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_loader = config.data.model_dump()
        self.generator = config.generator
        self.data_dir = config.data_dir
        self.fake = config.use_fake_dataset
        self.test_series=[]
        self.split = [config.training_set, config.validation_set, 1 - config.training_set - config.validation_set]
        self.predict = False

    def setup(self, stage):
        if stage == 'fit':
            train_series, val_series, self.test_series = self.get_random_series()
            self.train_dataset = RandomSeriesDataset(train_series, continuous_dynamics, self.generator)
            self.val_dataset = RandomSeriesDataset(val_series, continuous_dynamics, self.generator)
            print(f"Train: {len(self.train_dataset)} samples, val: {len(self.val_dataset)} samples")
        if stage == 'test':
            self.test_dataset = RandomSeriesDataset(self.test_series, continuous_dynamics, self.generator)
            print(f"Test: {len(self.test_dataset)} samples")
        if stage == 'predict':
            self.predict = True
            predict_series = self.get_random_series()
            self.predict_dataset = RandomSeriesDataset([predict_series], continuous_dynamics, self.generator)
            print(f"Predict: {len(self.predict_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.data_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.data_loader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.data_loader)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, shuffle=False, **self.data_loader)

    def get_random_series(self):
        if os.path.isfile('./data/dataset/series.npy'):
            series_list = load_subset('./data/dataset', end=self.generator.n_series)
        else:
            series_list = generate_time_series(**self.generator.get_generator_params())
        length = len(series_list)
        series_train = series_list[:int(length * self.split[0])]
        series_val = series_list[int(length * self.split[0]):int(length * (self.split[0]+self.split[1]))]
        series_test = series_list[int(length * (self.split[0]+self.split[1])):]
        if self.predict:
            return series_list[0]
        return series_train, series_val, series_test
