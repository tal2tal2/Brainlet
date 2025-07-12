import os

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from settings import Config
from utils.lds import generate_time_series
from utils.loaders import get_model_checkpoint
from utils.plots import time_series_prediction, evolution_with_background, dynamics_expert_assignment

if __name__ == '__main__':
    config = Config()
    config.predict()
    model = get_model_checkpoint(config.checkpoint, config.model)
    name = 'slds'
    if config.save_predictions:
        os.makedirs('./results/Figures', exist_ok=True)

    # Define simulation parameters
    gt_series, gt_states = generate_time_series(**config.generator.model_dump())[0]
    trainer = Trainer(**config.get_trainer_params())
    loader = DataLoader(gt_series, **config.data.model_dump())
    all_preds = trainer.predict(model, loader)

    derivative_list, direct_list, state_list = zip(*all_preds)
    pred_series_derivative = torch.cat(derivative_list, dim=0)
    pred_series_direct = torch.cat(direct_list, dim=0)
    pred_state = torch.cat(state_list, dim=0)

    time_series_prediction(gt_series, gt_states, pred_series_direct, pred_series_derivative, pred_state, name,
                           config.save_predictions)

    evolution_with_background(config.generator, gt_series, gt_states, pred_series_direct, pred_series_derivative,
                              pred_state, name, config.save_predictions)

    dynamics_expert_assignment(model, name, config.save_predictions)
