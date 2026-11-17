"""Shared image/label augmentation and normalization pipeline."""

import random

import albumentations as alb
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def make_tran(mode, seed=42):
    """Build the augmentation pipeline for ``mode`` (train / val / test)."""
    random.seed(seed)
    np.random.seed(seed)

    if mode == 'train':
        return alb.ReplayCompose([
            alb.RandomBrightnessContrast(p=1),
            alb.OneOf([
                alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                                     border_mode=cv2.BORDER_REFLECT, p=0.5),
                alb.GridDistortion(num_steps=3, distort_limit=0.05,
                                   border_mode=cv2.BORDER_REFLECT, p=0.3),
                alb.OpticalDistortion(distort_limit=0.05, shift_limit=0.05,
                                      border_mode=cv2.BORDER_REFLECT, p=0.3),
            ], p=0.5),
            alb.OneOf([
                alb.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                alb.RandomGamma(gamma_limit=(80, 120), p=0.5),
                alb.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.5),
            alb.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0),
            ToTensorV2()
        ])

    return alb.ReplayCompose([
        alb.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0),
        ToTensorV2()
    ])
