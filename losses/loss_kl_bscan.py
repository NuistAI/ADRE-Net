import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def _column_macro_iou_similarity_2d(label, class_ids=None, ignore_index=None, eps=1e-8):
    """
    Compute column-wise macro IoU similarity matrix for a single 2D label.
    label: [H, W], numpy array
    Returns: sim: [W, W], numpy float32
    """
    if class_ids is None:
        class_ids = np.unique(label).tolist()

    if ignore_index is not None:
        if isinstance(ignore_index, (list, tuple, set)):
            ignore_set = set(ignore_index)
        else:
            ignore_set = {ignore_index}
        class_ids = [c for c in class_ids if c not in ignore_set]

    sims_per_class = []
    for c in class_ids:
        X = (label == c).astype(np.float32)
        inter = X.T @ X
        area = X.sum(axis=0)
        union = area[:, None] + area[None, :] - inter
        sim_c = inter / (union + eps)
        sim_c = np.where(union > 0, sim_c, np.nan)
        sims_per_class.append(sim_c)

    if len(sims_per_class) == 0:
        W = label.shape[1]
        return np.zeros((W, W), dtype=np.float32)

    sims_per_class = np.stack(sims_per_class, axis=0)
    sim = np.nanmean(sims_per_class, axis=0)
    sim = np.nan_to_num(sim, nan=0.0)
    return sim.astype(np.float32)


def column_macro_iou_similarity(label, class_ids=None, ignore_index=None, eps=1e-8):
    """
    Compute column-wise macro IoU similarity matrix.
    label: [H, W] or [B, H, W], numpy or torch tensor
    Returns: sim: [W, W] or [B, W, W], numpy float32
    """
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
    label = np.asarray(label)

    if label.ndim == 2:
        return _column_macro_iou_similarity_2d(label, class_ids=class_ids, ignore_index=ignore_index, eps=eps)
    elif label.ndim == 3:
        results = []
        for i in range(label.shape[0]):
            sim = _column_macro_iou_similarity_2d(label[i], class_ids=class_ids, ignore_index=ignore_index, eps=eps)
            results.append(sim)
        return np.stack(results, axis=0)
    else:
        raise ValueError(f"label should be 2D [H,W] or 3D [B,H,W], got shape {label.shape}")


class AttentionLabelKLLoss(nn.Module):
    """
    Generate column similarity [B,S,S] from label [B,H,W] to supervise B-scan attention.

    Supports:
        1. Single layer: attn = [B,heads,S,S] or {"aux": {"BscanAttentionMap": ...}}
        2. Multi-layer list: attn = [[B,heads,S1,S1], [B,heads,S2,S2], ...]
           → compute column_sim per layer (label→resize→S×S), average the KL losses
    """

    def __init__(
        self,
        ignore_index=None,
        class_ids=None,
        eps=1e-8,
        remove_self=True,
        reduction="mean",
        per_head=False,
        layer_idx=None,
    ):
        """
        layer_idx: which attention layer(s) to supervise.
            - None: average over all layers (default)
            - int:  single layer index, e.g. -1 for the last layer, 0 for the first layer
            - list: specific layers, e.g. [-1] for the last layer only, [0, -1] for the first and last layers
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.class_ids = class_ids
        self.eps = eps
        self.remove_diag = remove_self
        self.reduction = reduction
        self.per_head = per_head
        self.layer_idx = layer_idx

        if self.reduction not in ["mean", "none"]:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
    def _column_weight(self, label):
        # label:
        # [B,H,W]
        if self.class_ids is None:
            return torch.ones(
                label.shape[0],
                label.shape[-1],
                device=label.device
            )
        # e.g. fluid id=1
        target_class = self.class_ids[-1]
        mask = (label == target_class).float()
        # fluid ratio per column
        ratio = mask.mean(dim=1)
        weight = 1.0 + ratio
        return weight
    def forward(self, attn, label):
        # ----------------------------------------
        # 0. dict → tensor / list
        # ----------------------------------------
        if isinstance(attn, dict):
            if "aux" in attn and "BscanAttentionMap" in attn["aux"]:
                attn = attn["aux"]["BscanAttentionMap"]
            elif "BscanAttentionMap" in attn:
                attn = attn["BscanAttentionMap"]
            else:
                return torch.tensor(0.0,
                    device=next(iter(attn.values()), torch.tensor(0.0)).device)

        if isinstance(label, dict):
            label = label.get("label", label)

        # ----------------------------------------
        # 1. list → select layers by layer_idx
        # ----------------------------------------
        if isinstance(attn, list):
            if self.layer_idx is not None:
                if isinstance(self.layer_idx, int):
                    idxs = [self.layer_idx]
                else:
                    idxs = list(self.layer_idx)
                selected = []
                for i in idxs:
                    i = i if i >= 0 else len(attn) + i
                    if 0 <= i < len(attn):
                        selected.append(attn[i])
                attn = selected if len(selected) > 0 else attn

            losses = [self._compute_layer(a, label) for a in attn]
            losses = [l for l in losses if l is not None]
            if not losses:
                return torch.tensor(0.0, device=label.device)
            return torch.stack(losses).mean()

        return self._compute_layer(attn, label)

    # ================================================================
    # single-layer computation
    # ================================================================
    def _compute_layer(self, attn, label):
        if not torch.is_tensor(attn):
            return attn.new_tensor(0.0) if torch.is_tensor(attn) else None

        # ---- multi-head handling ----
        num_repeat = 1  # number of times label must be repeated when per_head
        if attn.dim() == 5:
            # [B, W, heads, H, H]
            if self.per_head:
                B, W, H, S_h, S_w = attn.shape
                attn = attn.reshape(B * W * H, S_h, S_w)
                num_repeat = W * H
            else:
                attn = attn.mean(dim=2)  # [B, W, H, H]
        if attn.dim() == 4:
            # [B, heads, S, S]
            if self.per_head:
                B, H, S, S2 = attn.shape
                attn = attn.reshape(B * H, S, S2)
                num_repeat = H
            else:
                attn = attn.mean(dim=1)  # [B, S, S]
        elif attn.dim() != 3:
            return attn.new_tensor(0.0)

        B, S, S2 = attn.shape
        if S != S2:
            return attn.new_tensor(0.0)

        label = label.to(device=attn.device)

        # ---- first resize label to the attention map size, then compute similarity ----
        _, H_lab, W_lab = label.shape
        if (H_lab, W_lab) != (S, S):
            label = label.float().unsqueeze(1)              # [B, 1, H_lab, W_lab]
            label = F.interpolate(
                label, size=(S, S), mode='nearest',
            ).squeeze(1).long()                             # [B, S, S]
        if S< H_lab//2:
            self.ignore_index =None   
        sim_label_np = column_macro_iou_similarity(
            label, class_ids=self.class_ids,
            ignore_index=self.ignore_index, eps=self.eps,
        )
        sim_label = torch.from_numpy(sim_label_np).to(device=attn.device, dtype=attn.dtype)

        # ---- per_head: repeat label / sim_label to match the expanded batch ----
        if num_repeat > 1:
            sim_label = sim_label.repeat_interleave(num_repeat, dim=0)
            label = label.repeat_interleave(num_repeat, dim=0)

        # ---- remove diag ----
        if self.remove_diag:
            eye = torch.eye(S, device=attn.device, dtype=torch.bool).unsqueeze(0)
            attn = attn.masked_fill(eye, 0.0)
            sim_label = sim_label.masked_fill(eye, 0.0)

        # ---- normalization ----
        attn_prob = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        target_prob = sim_label / (sim_label.sum(dim=-1, keepdim=True) + self.eps)
        # ---- valid rows ----
        valid = sim_label.sum(dim=-1) > 0
        if valid.sum() == 0:
            return attn.new_tensor(0.0)

        # ---- KL(target || attn) ----
        kl = target_prob * (
            torch.log(target_prob + self.eps) - torch.log(attn_prob + self.eps)
        )
        kl_loss = kl.sum(dim=-1)
        weight = self._column_weight(label)
        weighted_loss = (
            kl_loss * weight
        )
        loss = weighted_loss[valid]
                

        return loss.sum() / (
            weight[valid].sum()+self.eps
        )
