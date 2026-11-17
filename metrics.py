import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __getitem__(self, index):
        return self.val[index]

    def get_na_num(self):
        if torch.is_tensor(self.val):
            isna = torch.isnan(self.val)
        else:
            isna = np.isnan(self.val)
        return isna.sum()

    def nan_to_zero(self):
        if torch.is_tensor(self.val):
            na_to_0 = torch.nan_to_num(self.val, 0.0)
        else:
            na_to_0 = np.nan_to_num(self.val)
        return na_to_0

    def update(self, val, n=1):
        self.val = val
        self.na_num = self.get_na_num()
        self.na_to_0 = self.nan_to_zero()
        self.sum += self.na_to_0.sum()
        self.count += n - self.na_num
        self.avg = self.sum / (self.count + 1e-20)

    def __str__(self):
        torch.set_printoptions(precision=5)
        return f'{self.val}'

    __repr__ = __str__


def dice_score(output, label, num_classes, valid_classes=None):
    """Per-class Dice coefficient for each valid class, shape [B, len(valid_classes)]."""
    if valid_classes is None:
        valid_classes = [1, 2, 3, 5, 6, 7]

    output = output.argmax(1).clone().detach()
    out_onehot = F.one_hot(output, num_classes=num_classes).permute(0, 3, 1, 2)
    lab_onehot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2)

    output_onehot = (out_onehot.data.cpu().numpy() == 1)
    label_onehot = (lab_onehot.data.cpu().numpy() == 1)

    intersections = (output_onehot & label_onehot)
    unions = (output_onehot | label_onehot)
    intersection = np.sum(intersections, (2, 3))
    union = np.sum(unions, (2, 3))

    smooth = 1e-5
    dice = (2 * intersection[:, valid_classes] + smooth) / \
           (union[:, valid_classes] + intersection[:, valid_classes] + smooth)
    return dice
