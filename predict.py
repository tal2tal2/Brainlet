import os

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from data.fake_data import RandomSeriesDataset
from settings import Config
from utils.lds import generate_time_series, continuous_dynamics
from utils.loaders import get_model_checkpoint
from utils.plots import time_series_prediction, evolution_with_background, dynamics_expert_assignment

if __name__ == '__main__':
    config = Config()
    config.predict()
    model = get_model_checkpoint(config.checkpoint, config.model, config.generator)
    name = 'slds'
    if config.save_predictions:
        os.makedirs('./results/Figures', exist_ok=True)

    # Define simulation parameters
    gt_series, gt_states = generate_time_series(**config.generator.get_generator_params())[0]
    dataset = RandomSeriesDataset([(gt_series, gt_states)],continuous_dynamics, config.generator)
    trainer = Trainer(**config.get_trainer_params())
    loader = DataLoader(dataset, **config.data.model_dump())
    all_preds = trainer.predict(model, loader)

    derivative_list, direct_list, state_list = zip(*all_preds)
    pred_series_derivative = torch.cat(derivative_list, dim=0)
    pred_series_direct = torch.cat(direct_list, dim=0)
    pred_state = torch.cat(state_list, dim=0)

    next_direct = pred_series_direct[:, 0, :]  # → (num_samples, D)
    next_derivative = pred_series_derivative[:, 0, :]  # → (num_samples, D)
    next_pred_state = pred_state[:, 0]  # → (num_samples,)
    # next_series = gt_series[:, 0]
    # next_state = gt_states[:, 0]
    time_series_prediction(gt_series, gt_states, next_direct, next_derivative, next_pred_state, name,
                           config.save_predictions)

    evolution_with_background(config.generator, gt_series, gt_states, next_direct, next_derivative,
                              next_pred_state, name, config.save_predictions)

    last_direct = pred_series_direct[:, -1, :]  # → (num_samples, D)
    last_derivative = pred_series_derivative[:, -1, :]  # → (num_samples, D)
    last_state = pred_state[:, -1]  # → (num_samples,)
    gt_series, gt_states = gt_series[config.generator.input_len+config.generator.target_len-1:], gt_states[config.generator.input_len+config.generator.target_len-1:]

    time_series_prediction(gt_series, gt_states, last_direct, last_derivative, last_state, name,
                           config.save_predictions)

    evolution_with_background(config.generator, gt_series, gt_states, last_direct, last_derivative,
                              last_state, name, config.save_predictions)

    # dynamics_expert_assignment(model, name, config.save_predictions)
