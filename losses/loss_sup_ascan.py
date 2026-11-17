import torch
import torch.nn as nn
import torch.nn.functional as F


class AscanSupConLossFromLabel(nn.Module):
    """
    Supervised Contrastive Loss for A-scan Attention.

    Supports two input forms:
        1. Single layer: qk_logits = [B,W,heads,H,H] or {"aux": {"AscanLogits": ...}}
        2. Multi-layer list: qk_logits = [[B,W,heads,H1,H1], [B,W,heads,H2,H2], ...]
           → resize label per layer and average the losses
    """

    def __init__(
        self,
        temperature=0.5,
        ignore_index=None,
        remove_self=True,
        sample_cols=None,
        eps=1e-8,
        per_head=True,
        layer_idx=None,
    ):
        """
        layer_idx: which attention layer(s) to supervise.
            - None: average over all layers (default)
            - int:  single layer index, e.g. -1 for the last layer, 0 for the first layer
            - list: specific layers, e.g. [-1] for the last layer only, [0, -1] for the first and last layers
        """
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.remove_self = remove_self
        self.sample_cols = sample_cols
        self.eps = eps
        self.per_head = per_head
        self.layer_idx = layer_idx

    def forward(self, qk_logits, label):
        # ----------------------------------------
        # 0. dict → tensor / list
        # ----------------------------------------
        if isinstance(qk_logits, dict):
            if "aux" in qk_logits and "AscanLogits" in qk_logits["aux"]:
                qk_logits = qk_logits["aux"]["AscanLogits"]
            elif "AscanLogits" in qk_logits:
                qk_logits = qk_logits["AscanLogits"]
            else:
                return torch.tensor(0.0,
                    device=next(iter(qk_logits.values()), torch.tensor(0.0)).device)

        if isinstance(label, dict):
            label = label.get("label", label)

        # ----------------------------------------
        # 1. list → select layers by layer_idx
        # ----------------------------------------
        if isinstance(qk_logits, list):
            if self.layer_idx is not None:
                if isinstance(self.layer_idx, int):
                    idxs = [self.layer_idx]
                else:
                    idxs = list(self.layer_idx)
                selected = []
                for i in idxs:
                    i = i if i >= 0 else len(qk_logits) + i
                    if 0 <= i < len(qk_logits):
                        selected.append(qk_logits[i])
                qk_logits = selected if len(selected) > 0 else qk_logits

            losses = [self._compute_layer(layer, label) for layer in qk_logits]
            losses = [l for l in losses if l is not None]
            if not losses:
                return torch.tensor(0.0, device=label.device)
            return torch.stack(losses).mean()

        # single layer (backward compatibility)
        return self._compute_layer(qk_logits, label)

    # ================================================================
    # single-layer computation: label is resized to the layer's attention map size
    # ================================================================
    def _compute_layer(self, qk_logits, label):
        # ---- 1. shape ----
        if qk_logits.dim() == 4:
            qk_logits = qk_logits.unsqueeze(2)            # [B,W,H,H] → [B,W,1,H,H]

        B, W, heads, H, H2 = qk_logits.shape
        assert H == H2

        # ---- 2. resize label to (H, W) ----
        label = label.to(qk_logits.device)

        if label.shape[-2:] != (H, W):
            label = label.float().unsqueeze(1)
            label = F.interpolate(label, size=(H, W), mode='nearest'
            ).squeeze(1).long()

        if label.shape[1] == H:
            label = label.permute(0, 2, 1)

        label = label.long()

        # ---- 3. sample columns ----
        if self.sample_cols is not None and self.sample_cols < W:
            idx = torch.randperm(W, device=qk_logits.device)[:self.sample_cols]
            qk_logits = qk_logits[:, idx]
            label = label[:, idx]
            W = self.sample_cols

        # ---- 4. flatten ----
        if self.per_head:
            # supervise each head independently (default)
            qk_logits = qk_logits.reshape(B * W * heads, H, H)
            label = label.unsqueeze(2).expand(B, W, heads, H).reshape(B * W * heads, H)
        else:
            # first average over heads, then compute the loss
            qk_logits = qk_logits.mean(dim=2)           # [B, W, H, H]
            qk_logits = qk_logits.reshape(B * W, H, H)
            label = label.reshape(B * W, H)

        # ---- 5. valid mask ----
        valid = torch.ones_like(label, dtype=torch.bool)
        if self.ignore_index is not None:
            if isinstance(self.ignore_index, (list, tuple, set)):
                for idx in self.ignore_index:
                    valid &= label != idx
            else:
                valid &= label != self.ignore_index

        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

        # ---- 6. positive mask ----
        same_label = label.unsqueeze(2) == label.unsqueeze(1)
        pos_weight = same_label.float() * valid_pair.float()

        if self.remove_self:
            eye = torch.eye(H, device=qk_logits.device, dtype=torch.bool).unsqueeze(0)
            pos_weight = pos_weight.masked_fill(eye, 0)
            valid_pair = valid_pair & (~eye)

        # ---- 7. log prob ----
        logits = qk_logits / self.temperature
        logits = logits.masked_fill(~valid_pair, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # ---- 8. SupCon loss ----
        pos_sum = pos_weight.sum(dim=-1)
        valid_query = pos_sum > 0

        if valid_query.sum() == 0:
            return qk_logits.new_tensor(0.0)

        loss_query = -(log_prob * pos_weight).sum(dim=-1)
        loss_query = loss_query / (pos_sum + self.eps)

        # ---- 9. class balanced ----
        label_flat = label
        class_losses = []
        class_weights = []

        alpha = 0.5   # recommended 0.3~0.7

        classes = torch.unique(label_flat[valid_query])

        for c in classes:

            class_mask = (label_flat == c) & valid_query

            if class_mask.sum() > 0:

                loss_c = loss_query[class_mask].mean()

                # number of samples in the class
                num_c = class_mask.sum().float()

                weight = 1.0 / (num_c ** alpha)

                class_losses.append(loss_c)
                class_weights.append(weight)


        class_losses = torch.stack(class_losses)
        class_weights = torch.stack(class_weights)


        class_weights = class_weights / (
            class_weights.sum()+self.eps
        )


        return (class_losses * class_weights).sum()
        '''class_losses = []

        for c in torch.unique(label_flat[valid_query]):
            class_mask = (label_flat == c) & valid_query
            if class_mask.sum() > 0:
                class_losses.append(loss_query[class_mask].mean())

        if not class_losses:
            return qk_logits.new_tensor(0.0)

        return torch.stack(class_losses).mean()'''
