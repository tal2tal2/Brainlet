import argparse
import gc
import os
import random
import re
from itertools import product

import numpy as np
import pandas as pd
import torch

from data.datamodule import DataModule
from settings import Config
from train import train


def append_results(test_results, id, model_path, results_path):
    epoch = int(re.search(r"epoch=(\d+)", model_path).group(1))

    # Collect info into a row
    result_row = {
        "run_id": id,
        "lambda_phys": lambda_phys,
        "lambda_peaky": lambda_peaky,
        "lambda_diverse": lambda_diverse,
        "epoch_end": epoch,
        "test_accuracy_t_0": test_results.get("test_accuracy_t_0", None),
        "test_accuracy_t_end": test_results.get("test_accuracy_t_end", None),
        "test_loss": test_results.get("test_loss", None),
        "test_loss_physics": test_results.get("test_loss_physics", None),
        "test_loss_peaky": test_results.get("test_loss_peaky", None),
        "test_loss_diverse": test_results.get("test_loss_diverse", None),
        "test_loss_MSE": test_results.get("test_loss_mse", None),
        "test_gating_sparsity": test_results.get("test_gating_sparsity", None),
        "test_gating_entropy": test_results.get("test_gating_entropy", None),
        "test_expert_0_usage": test_results.get("test_expert_0_usage", None),
        "test_expert_1_usage": test_results.get("test_expert_1_usage", None),
        "test_expert_2_usage": test_results.get("test_expert_2_usage", None),
        "test_MSE": test_results.get("test_MSE", None),
        "test_MAE": test_results.get("test_MAE", None),
        "test_R2_t_0": test_results.get("test_R2_t_0", None),
        "test_R2_t_end": test_results.get("test_R2_t_end", None),
        "model_path": model_path,
    }

    exist = os.path.exists(results_path)
    row_df = pd.DataFrame([result_row])

    with open(results_path, "a", newline='') as f:
        row_df.to_csv(f, index=False, header=not exist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sweep_length = 8

    config = Config()
    datamodule = DataModule(config)
    results_path = f"./sweep_results{args.gpu}.csv"

    parameters = {
        "lambda_phys": np.linspace(0.01, 5, sweep_length),
        "lambda_peaky": np.linspace(0.01, 0.3, sweep_length),
        "lambda_diverse": np.linspace(0.01, 0.3, sweep_length),
    }
    sweep_values = list(product(
        parameters["lambda_phys"],
        parameters["lambda_peaky"],
        parameters["lambda_diverse"]
    ))
    random.shuffle(sweep_values)
    total_jobs = len(sweep_values)
    jobs_per_rank = total_jobs // args.world_size
    start = args.rank * jobs_per_rank
    end = start + jobs_per_rank if args.rank < args.world_size - 1 else total_jobs


    print(f"GPU {args.gpu} handling jobs {start} to {end}")
    for i, (lambda_phys, lambda_peaky, lambda_diverse) in enumerate(sweep_values[start:end], start=start):
        print(f"\n=== Sweep {i + 1}/{total_jobs} ===")
        config.model.lambda_phys = lambda_phys
        config.model.lambda_peaky = lambda_peaky
        config.model.lambda_diverse = lambda_diverse

        test_results, id = train(config, datamodule)
        model_path = config.model_cb.best_model_path
        append_results(test_results, id, model_path, results_path)

        torch.cuda.empty_cache()
        gc.collect()
