"""HCMS (JHU) data loader.

``img_dir`` and ``lab_dir`` are the absolute paths to the RGB B-scans and the
palette label maps (same filenames), respectively.

The raw labels use ``0`` for both the air above the retina and the region below
the retina; ``layer_label`` reassigns the bottom background to class 9.
"""

import os

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image

from dataset.transforms import make_tran


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_dir, lab_dir,
                 data_type, resize=[256, 256], mode=None, seed=42):
        _inputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs")
        csv_file = os.path.join(_inputs_dir, dataset)
        self.img_ids = pd.read_csv(os.path.join(csv_file, f'{data_type}_images.txt'), index_col=None, header=None)[0]

        self.output_size = resize
        self.mode = mode
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.prep_tra_label = alb.Resize(p=1, height=resize[0], width=resize[1], interpolation=cv2.INTER_NEAREST)
        self.prep_tran_img = alb.Resize(p=1, height=resize[0], width=resize[1], interpolation=cv2.INTER_CUBIC)
        self.transform = make_tran(mode, seed)

    def __len__(self):
        return len(self.img_ids)

    def load_img_label(self, img_id):
        img_0 = np.array(Image.open(os.path.join(self.img_dir, img_id)).convert('RGB'))
        lab_0 = np.array(Image.open(os.path.join(self.lab_dir, img_id)).convert('P'))

        label = self.prep_tra_label(image=lab_0)['image']
        image_resize = self.prep_tran_img(image=img_0)['image']

        lab_list = {'image': image_resize, 'mask': label}
        lab_pics = self.transform(**lab_list)

        image = lab_pics['image'].float()
        label = lab_pics['mask'].long()

        return img_0, torch.from_numpy(lab_0).long(), image, label

    def layer_label(self, x):
        # Reassign the all-zero region below the retina to the bottom background (9).
        binary = (x > 0).long()
        cumsum = torch.cumsum(binary.flip(0), dim=0).flip(0)
        return torch.where(cumsum == 0, torch.tensor(9, dtype=x.dtype, device=x.device), x)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image_0, lab_0, image, child_lab = self.load_img_label(img_id)
        child_lab = self.layer_label(child_lab)
        lab_0 = self.layer_label(lab_0)
        # HCMS has no fluid classes, so the "layer-only" label equals the label.
        Layer_single_lab = child_lab
        return {'image': image, 'label': child_lab,
                'Layer_single_lab': Layer_single_lab,
                'image_0': image_0, 'lab_0': lab_0,
                'img_id': os.path.splitext(img_id)[0]}
