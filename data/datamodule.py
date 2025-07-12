from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from data.fake_data import RandomSeriesDataset
from utils.lds import generate_time_series, continuous_dynamics


class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_loader = config.data.model_dump()
        self.generator = config.generator
        self.data_dir = config.data_dir
        self.fake = config.use_fake_dataset

        self.split = [config.training_set, config.validation_set, 1 - config.training_set - config.validation_set]

    def setup(self, stage):
        if stage == 'fit':
            dataset = self.get_random_series()
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, self.split)
            print(f"Train: {len(self.train_dataset)} samples, val: {len(self.val_dataset)} samples")
        if stage == 'test':
            print(f"Test: {len(self.test_dataset)} samples")
        if stage == 'predict':
            self.dataset_predict = self.get_random_series()
            print(f"Predict: {len(self.dataset_predict)} samples")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.data_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.data_loader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.data_loader)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, shuffle=False, **self.data_loader)

    def get_random_series(self):
        series_list = generate_time_series(**self.generator.model_dump())
        dataset = RandomSeriesDataset(series_list, continuous_dynamics, self.generator.dt)
        return dataset
