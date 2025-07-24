import os
import random

import torch
from lightning import Trainer

from data.datamodule import DataModule
from models.moe.MoE import MixtureOfExperts
from settings import Config


def seed_setup(random_seed):
    os.environ['PYTHONHASHSEED'] = f'{random_seed}'
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: Config):
    import wandb
    try:
        seed_setup(config.random_seed)
        wandb.init(project='Brainlet', config=config.get_wandb_config())
        datamodule = DataModule(config)

        # Initialize model, optimizer, and loss function
        model = MixtureOfExperts(config.model, config.generator)

        trainer = Trainer(**config.get_trainer_params())
        trainer.fit(model, datamodule)
        model.eval()
        test_results = trainer.test(model, datamodule, verbose=False)[0]  # get first result dict
        return test_results, wandb.run.id
    finally:
        wandb.finish()


if __name__ == '__main__':
    train(Config())
