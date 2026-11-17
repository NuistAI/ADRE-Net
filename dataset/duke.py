"""Duke DME data loader.

``img_dir``, ``lab_dir`` and ``layer_lab_dir`` are the absolute paths to the RGB
B-scans, the complete label maps, and the layer-only label maps (same filenames),
respectively.

The complete label (``lab_dir``) uses ``0`` for the air and the region below the
retina, ``1``-``7`` for the layers, and ``8`` for the fluid. The loader remaps it
to the model scheme: fluid ``8`` -> ``9``, then the below-retina ``0`` region is
filled with the bottom background ``8``.

The layer-only label (``layer_lab_dir``, values ``0``-``7``) is loaded as the
anatomical "single-layer" target; the below-retina ``0`` region is likewise
filled with ``8``.
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
    def __init__(self, dataset, img_dir, lab_dir, layer_lab_dir,
                 data_type, resize=[256, 256], mode=None, seed=42):
        _inputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs")
        csv_file = os.path.join(_inputs_dir, dataset)
        self.img_ids = pd.read_csv(os.path.join(csv_file, f'{data_type}_images.txt'), index_col=None, header=None)[0]

        self.output_size = resize
        self.mode = mode
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.layer_lab_dir = layer_lab_dir
        self.prep_tra_label = alb.Resize(p=1, height=resize[0], width=resize[1], interpolation=cv2.INTER_NEAREST)
        self.prep_tran_img = alb.Resize(p=1, height=resize[0], width=resize[1], interpolation=cv2.INTER_CUBIC)
        self.transform = make_tran(mode, seed)

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def _fill_bottom_background(x):
        # Fill the all-zero region below the retina with the bottom background (8).
        binary = (x > 0).long()
        cumsum = torch.cumsum(binary.flip(0), dim=0).flip(0)
        return torch.where(cumsum == 0, torch.tensor(8, dtype=x.dtype, device=x.device), x)

    def layer_label(self, x):
        # Complete label: fluid 8 -> 9, then fill the below-retina background as 8.
        x = x.clone()
        x[x == 8] = 9
        return self._fill_bottom_background(x)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_0 = np.array(Image.open(os.path.join(self.img_dir, img_id)).convert('RGB'))
        lab_0 = np.array(Image.open(os.path.join(self.lab_dir, img_id)).convert('P'))
        layer_lab_0 = np.array(Image.open(os.path.join(self.layer_lab_dir, img_id)).convert('P'))

        image_resize = self.prep_tran_img(image=img_0)['image']
        label_resize = self.prep_tra_label(image=lab_0)['image']
        layer_label_resize = self.prep_tra_label(image=layer_lab_0)['image']

        # Augment image + complete label, then replay the same transform on the layer-only label.
        transformed = self.transform(image=image_resize, mask=label_resize)
        image = transformed['image'].float()
        label = self.layer_label(transformed['mask'].long())
        layer_transformed = alb.ReplayCompose.replay(
            transformed['replay'], image=image_resize, mask=layer_label_resize)
        Layer_single_lab = self._fill_bottom_background(layer_transformed['mask'].long())

        lab_0 = self.layer_label(torch.from_numpy(lab_0).long())

        return {'image': image, 'label': label,
                'Layer_single_lab': Layer_single_lab,
                'image_0': img_0, 'lab_0': lab_0,
                'img_id': os.path.splitext(img_id)[0]}
