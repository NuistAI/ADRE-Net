"""Entry point for training and evaluating ADRE-Net.

Usage
-----
    python main.py \
        --dataset aroi \
        --img_dir /path/to/AROI/image \
        --lab_dir /path/to/AROI/label \
        --outputs_dir /path/to/outputs \
        --backbone adre_net \
        --loss ce_dice_wbce_asup_bkl

The script trains the model, evaluates the best checkpoint and writes the
per-class Dice plus the anatomy-consistency metrics (MAD, layer fragmentation
incidence/severity, layer-order violation incidence/severity, fluid
recall/precision).
"""

import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml

from adre_net import choice_backbone
from configs import parse_args
from evaluate import main_test
from trainer import train





def get_config(dataset):
    return vars(parse_args(dataset))


def build_model(config):
    """Instantiate ADRE-Net from scratch."""
    return choice_backbone(config["backbone"], config["num_classes"], input_config=config)


def set_save_path(config):
    save_output_path = config["outputs_dir"]

    os.makedirs(os.path.join(save_output_path, "checkpoint"), exist_ok=True)

    config_file = os.path.join(save_output_path, "config.yml")
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            yaml.dump(config, f)

    return save_output_path


def run_evaluation(config, save_path, model, is_draw=True):
    """Evaluate the best checkpoint and write the summary metrics."""
    best_path = os.path.join(save_path, "checkpoint", "best.pth")
    if not os.path.exists(best_path):
        print("No best model found.")
        return 0.0

    print("Best model:", best_path)
    best_result = main_test(config, save_path, model,
                            dataset_type="test", pre_model_path="best.pth",
                            is_draw=is_draw)

    df = pd.DataFrame(best_result)
    df["dataset_type"] = "test"

    class_names = config['class_names']
    mad_boundary_cols = [k for k in best_result if k.startswith("mad_b")]
    summary_cols = ["dice"] + class_names + ["mad"] + mad_boundary_cols + \
                   ["lfi", "lfs", "lovi", "lovs"]
    if config['fluid_ids']:
        summary_cols += ["fluid_precision", "fluid_recall"]

    new_csv_path = os.path.join(save_path, "best_result.csv")
    new_sub = df[summary_cols + ["dataset_type"]]
    write_header = not os.path.exists(new_csv_path)
    with open(new_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(list(new_sub.columns))
        for _, row in new_sub.iterrows():
            writer.writerow(row.tolist())

    return best_result["dice"]

def get_dataset():
    """Extract ``--dataset`` from the command line for dispatch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="aroi")
    args, _ = parser.parse_known_args()
    return args.dataset

if __name__ == "__main__":
    config = get_config(get_dataset())
    cudnn.benchmark = False
    cudnn.deterministic = True

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("-" * 40)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 40)

    save_path = set_save_path(config)
    model = build_model(config)
    model.cuda()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    train(config, save_path, model)
    run_evaluation(config, save_path, model, is_draw=True)
