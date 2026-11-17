import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None, device='cpu'):
        super(DiceLoss, self).__init__()
        self.device = device
        self.n_classes = n_classes
        if weight is None:
            weight = torch.ones(n_classes)
        else:
            if not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight, dtype=torch.float32)
            if weight.numel() != n_classes:
                raise ValueError(f"weight length ({weight.numel()}) != n_classes ({n_classes})")
        self.register_buffer('weight', weight)

    def forward(self, outputs, labels):
        if isinstance(labels, dict):
            labels = labels['label'].to(self.device)
        if isinstance(outputs, dict):
            outputs = outputs['segment_output'].to(self.device)
        # auto-resize label to match prediction size
        if outputs.shape[-2:] != labels.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=outputs.shape[-2:],
                mode='nearest').squeeze(1).long()
        outputs = F.softmax(outputs, dim=1)
        labels = F.one_hot(labels, num_classes=self.n_classes).permute((0, 3, 1, 2))

        loss = 0.0
        total_w = 0.0
        for i in range(self.n_classes):
            dice = self._dice(outputs[:, i], labels[:, i])
            loss += self.weight[i] * dice
            total_w += self.weight[i]
        return loss / total_w

    @staticmethod
    def _dice(outputs, labels, smooth=1e-5):
        outputs, labels = outputs.reshape(-1), labels.reshape(-1)
        inter = (outputs * labels).sum()
        union = (outputs + labels).sum()
        return 1 - (2 * inter + smooth) / (union + smooth)
