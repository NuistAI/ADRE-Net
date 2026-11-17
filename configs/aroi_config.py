"""AROI experiment configuration (argparse).

Holds the hyper-parameters used in the ADRE-Net paper for the AROI dataset:
joint segmentation of 3 retinal layers + 3 fluids (8 classes) at 256x256 with
AdamW, ReduceLROnPlateau and a combined anatomical loss (CE + Dice + L_ARA +
L_AttnCon + L_AttnKL).
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ADRE-Net (AROI)")

    # device / reproducibility
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_gpu", type=int, default=1)

    # training loop
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--early_stopping", default=30, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # backbone / loss
    parser.add_argument("--backbone", default="adre_net")
    parser.add_argument("--loss", default="ce_dice_wbce_asup_bkl")
    parser.add_argument("--ignore_index", default=None, nargs="*", type=int,
                        help="class ids to ignore in the attention losses (default: ignore none)")

    # dataset
    parser.add_argument("--dataset", default="aroi")
    parser.add_argument("--num_classes", default=8, type=int)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--image_height", default=256, type=int)
    parser.add_argument("--image_width", default=256, type=int)
    parser.add_argument("--pixel_spacing", default=1.96, type=float,
                        help="physical pixel size in um/pixel (used to report MAD in um)")
    parser.add_argument("--img_dir", default="/path/to/AROI/image",
                        help="absolute path to the AROI RGB B-scan folder")
    parser.add_argument("--lab_dir", default="/path/to/AROI/label",
                        help="absolute path to the AROI label folder")
    parser.add_argument("--class_weight", default=None)

    # anatomical prior: class names, layer order, layer/fluid ids
    parser.add_argument("--class_names", default=["ILM-IPL/INL", "IPL/INL-RPE", "RPE-BM", "PED", "SRF", "IRF"], nargs="*",
                        help="names of the valid (non-background) classes, ordered by class id")
    parser.add_argument("--contain_edges", default=[(7, 1), (7, 2), (5, 3), (6, 2)])
    parser.add_argument("--directed_edges", default=[(0, 1), (1, 2), (2, 3), (3, 4)])
    parser.add_argument("--layer_ids", default=[0, 1, 2, 3, 4], nargs="*", type=int,
                        help="layer/background class ids (top bg + 3 layers + bottom bg)")
    parser.add_argument("--fluid_ids", default=[5, 6, 7], nargs="*", type=int,
                        help="fluid/lesion class ids (PED / SRF / IRF)")
    parser.add_argument("--valid_classes", default=[1, 2, 3, 5, 6, 7], nargs="*", type=int,
                        help="class ids included in evaluation metrics (exclude background)")

    # paths
    parser.add_argument("--outputs_dir", default="./outputs")

    # optimizer / scheduler
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--scheduler", default="ReduceLROnPlateau")
    parser.add_argument("--lr", "--learning_rate", dest="lr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=5e-6, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--scheduler_factor", default=0.8, type=float,
                        help="ReduceLROnPlateau factor")
    parser.add_argument("--scheduler_patience", default=3, type=int,
                        help="ReduceLROnPlateau patience")
    parser.add_argument("--scheduler_cooldown", default=3, type=int,
                        help="ReduceLROnPlateau cooldown")

    return parser.parse_args()
