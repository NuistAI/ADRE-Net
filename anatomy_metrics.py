import numpy as np
import torch
import torch.nn.functional as F


def extract_boundary(mask, boundary="top"):
    """
    Extract the per-column boundary ordinate from a binary mask.

    mask: [H, W] numpy bool array
    boundary: "top" takes the minimum row coordinate per column, "bottom" the maximum
    Returns: [W] per-column boundary coordinate, or nan where the layer is absent
    """
    W = mask.shape[1]
    coords = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        col = np.where(mask[:, x])[0]
        if len(col) == 0:
            continue
        coords[x] = col.min() if boundary == "top" else col.max()
    return coords


def mad_between(pred_mask, gt_mask, boundary="top", ignore_nan=True):
    """
    Compute MAD for a single boundary (unit: pixel).

    pred_mask, gt_mask: [H, W] numpy bool arrays
    boundary: "top" or "bottom"
    ignore_nan: if True, skip columns missing in prediction or ground truth
    """
    pred_b = extract_boundary(pred_mask, boundary)
    gt_b = extract_boundary(gt_mask, boundary)

    if ignore_nan:
        valid = ~np.isnan(pred_b) & ~np.isnan(gt_b)
        if valid.sum() == 0:
            return np.nan
        return np.abs(pred_b[valid] - gt_b[valid]).mean()
    else:
        return np.nanmean(np.abs(pred_b - gt_b))


def mad_metrics(output, label, num_classes, boundary_spec, pixel_spacing=1.0):
    """
    Compute MAD.

    output: logits [B, num_classes, H, W]
    label:  [B, H, W] integer class map
    num_classes: total number of classes
    boundary_spec: list of boundaries for MAD, each element is (class_id, "top"/"bottom")
    pixel_spacing: physical size per pixel (um/pixel)

    Returns:
        mad_per_boundary: [B, len(boundary_spec)]
        mad_mean:         [B] per-image MAD averaged over boundaries
    """

    output = output.argmax(1).clone().detach()
    out_onehot = F.one_hot(output, num_classes=num_classes).permute(0, 3, 1, 2)
    lab_onehot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2)

    output_onehot = (out_onehot.data.cpu().numpy() == 1)  # [B, C, H, W]
    label_onehot = (lab_onehot.data.cpu().numpy() == 1)

    B = output_onehot.shape[0]
    mad_all = []

    for (c, side) in boundary_spec:
        mad_boundary = []
        for b in range(B):
            pred_mask = output_onehot[b, c]
            gt_mask = label_onehot[b, c]
            mad = mad_between(pred_mask, gt_mask, boundary=side)
            mad_boundary.append(mad * pixel_spacing if not np.isnan(mad) else np.nan)
        mad_all.append(mad_boundary)

    mad_per_boundary = np.array(mad_all).T          # [B, len(boundary_spec)]
    mad_mean = np.nanmean(mad_per_boundary, axis=1)  # [B]

    return mad_per_boundary, mad_mean


def fragmentation_count(output, layer_ids, fluid_ids):
    """
    Layer fragmentation count.

    output: logits [B, num_classes, H, W]
    layer_ids: retinal layer class indices (e.g. [1..7])
    fluid_ids: fluid class index (int) or index list (list/tuple), e.g. 9 or [5,6,7]

    Returns:
        per_img_total: [B] sum of fragment counts over all layers (connected components - 1 per layer, accumulated)
        per_img_flag:  [B] whether any fragment exists in the image (0/1)
    """
    import cv2
    import numpy as np

    pred = output.argmax(1).cpu().numpy()  # [B,H,W]
    if isinstance(fluid_ids, (list, tuple)):
        fluid = np.isin(pred, fluid_ids)
    else:
        fluid = (pred == fluid_ids)
    B = pred.shape[0]

    per_img_total = np.zeros(B, dtype=np.float32)
    per_img_flag = np.zeros(B, dtype=np.float32)

    for b in range(B):
        total = 0
        for lid in layer_ids:
            layer = (pred[b] == lid)
            if layer.sum() == 0:
                continue

            union = (layer | fluid[b]).astype(np.uint8)
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
            if num_labels <= 1:
                continue

            comp_areas = np.array([
                np.count_nonzero((labels == c) & layer)
                for c in range(1, num_labels)
            ])
            comp_areas = comp_areas[comp_areas > 0]  # keep only connected components that truly contain the layer

            if len(comp_areas) <= 1:
                continue

            total += len(comp_areas) - 1

        per_img_total[b] = float(total)
        per_img_flag[b] = 1.0 if total > 0 else 0.0

    return per_img_total, per_img_flag


def layer_cross_stats(output, layer_ids):
    """
    Layer-order crossing statistics:
        has_cross:    [B] whether the image has at least one layer-order crossing, 0/1
        cross_pixels: [B] total number of crossing pixels in the image
        valid_pixels: [B] total comparable layer-pair pixels in the image (optional, for normalization)

    output: logits [B, num_classes, H, W]
    layer_ids: retinal layer class indices in anatomical top-to-bottom order (e.g. [1..7])
    """
    pred = output.argmax(1)  # [B,H,W]
    H = pred.shape[1]
    labels = torch.as_tensor(layer_ids, device=pred.device)
    L = len(labels)

    rows = torch.arange(H, device=pred.device)[:, None]  # [H,1]
    eq = pred[:, None] == labels[None, :, None, None]    # [B,L,H,W]
    top = torch.where(eq, rows[None, None], H).amin(dim=2)      # [B,L,W]
    bottom = torch.where(eq, rows[None, None], -1).amax(dim=2)  # [B,L,W]
    valid = top < H  # [B,L,W]

    i, j = torch.triu_indices(L, L, 1, device=pred.device)
    valid_pair = valid[:, i] & valid[:, j]                    # [B,npair,W]

    # Class-j pixels located above or overlapping the bottom of class i
    rows_b = rows[None, None]                                 # [1,1,H,1]
    bottom_i = bottom[:, i][:, :, None, :]                    # [B,npair,1,W]
    eq_j = eq[:, j]                                           # [B,npair,H,W]
    cross = (eq_j & (rows_b <= bottom_i)).sum(dim=2)          # [B,npair,W]
    cross = cross * valid_pair                                # only count columns where both layers are valid

    cross_pixels = cross.sum(dim=(1, 2)).float()              # [B]
    has_cross = (cross_pixels > 0).float()                    # [B]
    valid_pixels = valid_pair.sum(dim=(1, 2)).float() * H     # [B] optional

    return has_cross.cpu().numpy(), cross_pixels.cpu().numpy(), valid_pixels.cpu().numpy()


def fluid_metrics(output, label, fluid_ids):
    """
    Per-image binary metrics for fluid (pathology): precision and recall.
    Multiple fluid classes are merged into a single category and computed as a whole.

    output: logits [B, num_classes, H, W]
    label:  ground-truth label [B, H, W]
    fluid_ids: fluid class indices (int or list/tuple), e.g. [5, 6, 7]

    Returns:
        precision [B]: TP / (TP + FP)
        recall    [B]: TP / (TP + FN)
    """
    pred = output.argmax(1)  # [B,H,W]

    if isinstance(fluid_ids, (list, tuple)):
        ids = torch.as_tensor(fluid_ids, device=pred.device)
        fluid_pred = torch.isin(pred, ids)
        fluid_gt = torch.isin(label, ids)
    else:
        fluid_pred = (pred == fluid_ids)
        fluid_gt = (label == fluid_ids)

    tp = (fluid_pred & fluid_gt).flatten(1).sum(1).float()      # [B]
    fp = (fluid_pred & ~fluid_gt).flatten(1).sum(1).float()
    fn = (~fluid_pred & fluid_gt).flatten(1).sum(1).float()

    smooth = 1e-5
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision.cpu().numpy(), recall.cpu().numpy()


def hierarchical_decode(logits, layer_ids, fluid_ids, contain_edges, tau=0.5):
    """
    Anatomical-prior-constrained hierarchical decoding (inference only).

    At each pixel, the predicted layer is used as an anatomical prior to restrict
    fluid candidates. Among the fluid classes compatible with the predicted layer
    (via ``contain_edges``, which encodes the fluid-layer relation R_LF), those
    whose sigmoid confidence exceeds ``tau`` are kept, and the argmax of their
    logits is chosen. If no compatible fluid candidate passes the threshold, the
    predicted layer is output.

    logits:        [B, C, H, W] network logits
    layer_ids:     layer/background class ids (C_layer)
    fluid_ids:     fluid class ids
    contain_edges: list of (fluid, layer) pairs encoding R_LF
    tau:           fluid acceptance threshold (sigmoid), 0.5 in the paper

    Returns: final integer label [B, H, W]
    """
    device = logits.device
    layer_ids = torch.as_tensor(layer_ids, device=device)
    fluid_ids = torch.as_tensor(fluid_ids, device=device)

    # Layer-only datasets (no fluids / no containment relations): plain argmax over layers.
    if len(fluid_ids) == 0 or len(contain_edges) == 0:
        return layer_ids[logits[:, layer_ids].argmax(dim=1)]

    L, F = len(layer_ids), len(fluid_ids)

    # fluid-layer compatibility matrix [L, F]
    compat = torch.zeros(L, F, dtype=torch.bool, device=device)
    for fluid, layer in contain_edges:
        fi = (fluid_ids == fluid).nonzero(as_tuple=True)[0]
        li = (layer_ids == layer).nonzero(as_tuple=True)[0]
        if len(fi) and len(li):
            compat[li, fi] = True

    layer_logits = logits[:, layer_ids]          # [B, L, H, W]
    hat_l_idx = layer_logits.argmax(dim=1)        # [B, H, W]

    fluid_logits = logits[:, fluid_ids]           # [B, F, H, W]
    fluid_prob = torch.sigmoid(fluid_logits)      # [B, F, H, W]

    compat_pp = compat[hat_l_idx].permute(0, 3, 1, 2)   # [B, F, H, W]
    cand_mask = compat_pp & (fluid_prob > tau)          # [B, F, H, W]

    masked_fluid = fluid_logits.masked_fill(~cand_mask, float('-inf'))
    best_fluid_idx = masked_fluid.argmax(dim=1)         # [B, H, W]
    has_candidate = cand_mask.any(dim=1)                # [B, H, W]

    fluid_label = fluid_ids[best_fluid_idx]
    layer_label = layer_ids[hat_l_idx]
    return torch.where(has_candidate, fluid_label, layer_label)
