import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class CELoss(nn.Module):
    def __init__(self, num_classes, class_weight=None, ignore_index=-100, device='cpu'):
        super(CELoss, self).__init__()
        self.device = device
        if class_weight is not None:
            weights_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
            self.ce_loss = CrossEntropyLoss(weight=weights_tensor, reduction='mean')
        else:
            self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes

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
        return self.ce_loss(outputs, labels)
