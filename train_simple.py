import os
import random

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import optim, nn
from torch.utils.data import DataLoader, random_split

from data.fake_data import FakeSeriesDataset
from models.moe.MoE import MixtureOfExperts
from utils.lds import generate_time_series, continuous_dynamics
random_seed = 660
os.environ['PYTHONHASHSEED'] = f'{random_seed}'
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    import wandb

    wandb.init(project='Brainlet')
    # Generate series and create dataset (using continuous dynamics here)
    series_list = generate_time_series(n_series=20, series_length=10000, dt=0.001, use_slds=False)
    dataset = FakeSeriesDataset(series_list, continuous_dynamics, dt=0.001)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True,prefetch_factor=2,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True,prefetch_factor=2,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True,prefetch_factor=2,
                             shuffle=False)

    # Initialize model, optimizer, and loss function
    model = MixtureOfExperts(input_dim=2, output_dim=4, num_experts=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    num_epochs = 5
    lambda_phys = 0.1
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True)
    trainer = Trainer(max_epochs=num_epochs, accumulate_grad_batches=16, callbacks=[model_checkpoint],
                      logger=WandbLogger())
    trainer.fit(model, train_loader, val_loader)
    model.eval()
    trainer.test(model, test_loader)
