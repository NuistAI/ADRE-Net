import os
import cv2
import torch
import numpy as np


def save_qk_logits_heatmap_cv2(
    qk_logits,
    save_dir,
    epoch,
    img_id=None,
    sample_idx=0,
    head_idx=None,
    prefix="qk_logits",
    resize_to=512,
    query_row=None,
):
    """
    Save a QK logits heatmap using OpenCV.

    Args:
        qk_logits:
            [B, H, S, S] or [B, S, S]
        save_dir:
            output folder
        epoch:
            current epoch
        img_id:
            current sample image name without suffix, e.g. "000123"
        sample_idx:
            index of the sample in the batch
        head_idx:
            None for multi-head average; int for a specific head.
        prefix:
            output filename prefix
        resize_to:
            output image size, default 512x512
        query_row:
            optional row index (in the source H) to mark with a green line
    """

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(qk_logits, dict):
        qk_logits = qk_logits["logits"]

    qk_logits = qk_logits.detach().float().cpu()

    if qk_logits.dim() == 4:
        # [B, H, S, S]
        if head_idx is None:
            mat = qk_logits[sample_idx].mean(dim=0)
            head_name = "mean_head"
        else:
            mat = qk_logits[sample_idx, head_idx]
            head_name = f"head_{head_idx}"

    elif qk_logits.dim() == 3:
        # [B, S, S]
        mat = qk_logits[sample_idx]
        head_name = "single_head"

    else:
        raise ValueError(f"Unsupported qk_logits shape: {qk_logits.shape}")

    mat = mat.numpy()
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    mat_min = mat.min()
    mat_max = mat.max()

    if mat_max - mat_min < 1e-8:
        mat_norm = np.zeros_like(mat, dtype=np.uint8)
    else:
        mat_norm = (mat - mat_min) / (mat_max - mat_min)
        mat_norm = (mat_norm * 255).astype(np.uint8)

    if resize_to is not None:
        mat_norm = cv2.resize(
            mat_norm,
            (resize_to, resize_to),
            interpolation=cv2.INTER_NEAREST,
        )

    heatmap = cv2.applyColorMap(mat_norm, cv2.COLORMAP_VIRIDIS)

    if query_row is not None:
        # Mark the query depth with a green line (mapped from the source H to the output size).
        src_h = mat.shape[0]
        dst_h = heatmap.shape[0]
        y = int(round(query_row * dst_h / src_h))
        y = min(max(y, 0), dst_h - 1)
        cv2.line(heatmap, (0, y), (heatmap.shape[1] - 1, y), (0, 255, 0), 1)

    if img_id is None:
        img_id = f"sample_{sample_idx}"

    # Avoid path separators in the filename
    img_id = str(img_id).replace("/", "_").replace("\\", "_")
    

    save_name = f"{img_id}_{head_name}_epoch_{epoch:03d}_{prefix}.png"
    save_path = os.path.join(save_dir, save_name)

    cv2.imwrite(save_path, heatmap)

    return save_path
