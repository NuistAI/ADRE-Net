import datetime
import gc
import logging
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.color_set import get_color_set
from dataset import get_dataset_class
from losses.default import build_losses, criterion
from metrics import AverageMeter, dice_score
from anatomy_metrics import hierarchical_decode
from utils.qk_visualization import save_qk_logits_heatmap_cv2
from visualize import draw_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Optimizer & scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer(config, model):
    """Create the AdamW optimizer and ReduceLROnPlateau scheduler."""
    param_groups = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.AdamW(param_groups, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'],
        patience=config['scheduler_patience'], cooldown=config['scheduler_cooldown'],
        min_lr=config['min_lr'])

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_float(x):
    if hasattr(x, "detach"):
        return x.detach().item()
    if hasattr(x, "item"):
        return x.item()
    return float(x)


# ---------------------------------------------------------------------------
# Training & validation (one epoch)
# ---------------------------------------------------------------------------

def run_epoch(config, train_loader, model, optimizer, losses, root_logger):
    avg_meters = {'train_dice': AverageMeter()}
    pbar = tqdm(total=len(train_loader))

    if config['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.train()

    for _, batch in enumerate(train_loader):
        image_batch = batch['image']
        label_batch = batch['label']
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        inputs = {'image': image_batch, 'label': label_batch}

        optimizer.zero_grad()
        outputs = model(inputs)

        losses = criterion(config, batch, outputs, losses=losses)
        total_loss = losses['total_loss']

        total_loss.backward()
        optimizer.step()

        seg_outputs = outputs['segment_output']
        if seg_outputs.shape[-2:] != label_batch.shape[-2:]:
            label_batch = F.interpolate(label_batch.unsqueeze(1).float(), size=seg_outputs.shape[-2:], mode='nearest').squeeze(1).long()
        dice_mean = dice_score(seg_outputs, label_batch, config['num_classes'], valid_classes=config['valid_classes'])

        for loss_name, loss_value in losses.items():
            meter_name = "train_loss" if loss_name == "total_loss" else f"train_{loss_name}"
            if meter_name not in avg_meters:
                avg_meters[meter_name] = AverageMeter()
            if hasattr(loss_value, "item"):
                loss_value = loss_value.item()
            avg_meters[meter_name].update(loss_value)

        avg_meters['train_dice'].update(np.mean(dice_mean))

        postfix = OrderedDict((n, m.avg) for n, m in avg_meters.items())
        head_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']

        loss_str = ", ".join(f"{n}: {to_float(v):.4f}" for n, v in losses.items())
        root_logger.info(f"head_lr:{head_lr}, Losses: {loss_str}, train_dice:{postfix.get('train_dice', 0):.4f}")

        pbar.set_postfix({'head_lr': head_lr, **postfix})
        pbar.update(1)

    pbar.close()
    return postfix


def _query_depth(lab_0, size, col):
    """Return the query depth (top edge of class 3) at column ``col``, at scale ``size``."""
    lab = cv2.resize(lab_0, (size, size), interpolation=cv2.INTER_NEAREST)
    rows_3 = np.where(lab[:, col] == 3)[0]
    return int(rows_3[0]) if len(rows_3) else size // 2


def _save_qk_context(batch, save_dir, epoch, config, size, col, q, scale_idx, display):
    """Downsample the original image/GT to the scale resolution, annotate the
    column and the class-2/3 boundary (query depth), then upsample to ``display``."""
    img_0 = batch["image_0"][0].cpu().numpy()      # [H0, W0, 3] RGB
    lab_0 = batch["lab_0"][0].cpu().numpy()        # [H0, W0]

    img = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    lab = cv2.resize(lab_0, (size, size), interpolation=cv2.INTER_NEAREST)
    gt = draw_mask(lab, get_color_set(config["dataset"]), n_class=config["num_classes"])

    for canvas in (img, gt):
        cv2.line(canvas, (col, 0), (col, size - 1), (0, 0, 255), 1)   # column (red)
        cv2.circle(canvas, (col, q), 2, (0, 255, 0), -1)              # query point (green dot)

    img = cv2.resize(img, (display, display), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt, (display, display), interpolation=cv2.INTER_NEAREST)

    out_dir = os.path.join(save_dir, "qk_vis", "context", f"scale_{scale_idx}", f"epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    img_id = batch["img_id"][0]
    cv2.imwrite(os.path.join(out_dir, f"{img_id}_orig.png"), img)
    cv2.imwrite(os.path.join(out_dir, f"{img_id}_gt.png"), gt)


def _save_qk_heatmaps(outputs, batch, save_dir, epoch, config):
    """Save the QK attention heatmaps and annotated context for all decoder scales."""
    aux = outputs.get("aux", {})
    if not aux:
        return
    img_id = batch["img_id"][0]
    display = config["image_height"]

    ascan_logits = aux.get("AscanLogits")     # list of 5 (coarse -> fine)
    bscan_logits = aux.get("BscanLogits")
    num_scales = len(ascan_logits) if ascan_logits else (len(bscan_logits) if bscan_logits else 0)

    for i in range(num_scales):
        size = col = q = None

        # A-scan (vertical) attention: middle column, averaged over heads.
        if ascan_logits:
            qk = ascan_logits[i]              # [B, W, heads, H, H]
            if qk.dim() == 5:
                w_idx = qk.shape[1] // 2
                size, col = qk.shape[-1], w_idx
                q = _query_depth(batch["lab_0"][0].cpu().numpy(), size, col)
                qk = qk[:, w_idx]             # [B, heads, H, H]
            save_qk_logits_heatmap_cv2(
                qk_logits=qk,
                save_dir=os.path.join(save_dir, "qk_vis", "AscanLogits", f"scale_{i}", f"epoch_{epoch}"),
                epoch=epoch, img_id=img_id, sample_idx=0, prefix=f"w{w_idx}",
                resize_to=display, query_row=q)

        # B-scan (horizontal) attention, averaged over heads.
        if bscan_logits:
            qk = bscan_logits[i]              # [B, heads, W, W]
            save_qk_logits_heatmap_cv2(
                qk_logits=qk,
                save_dir=os.path.join(save_dir, "qk_vis", "BscanLogits", f"scale_{i}", f"epoch_{epoch}"),
                epoch=epoch, img_id=img_id, sample_idx=0, prefix="bscan", resize_to=display)

        if size is not None:
            _save_qk_context(batch, save_dir, epoch, config, size, col, q, i, display)


def run_validation(config, val_loader, model, losses, root_logger, epoch=None, save_dir=None):
    class_names = config['class_names']
    avg_meters = {'val_dice': AverageMeter()}
    for name in class_names:
        avg_meters[name] = AverageMeter()

    model.eval()
    pbar = tqdm(total=len(val_loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            image_batch = batch['image'].to(device)
            label_batch = batch['label'].to(device)
            lab_0 = batch['lab_0'].to(config['device'])

            outputs = model({'image': image_batch, 'label': label_batch})
            if epoch is not None and epoch % 10 == 0 and batch_idx == 0:
                _save_qk_heatmaps(outputs, batch, save_dir, epoch, config)
            losses = criterion(config, batch, outputs, losses=losses)
            seg_outputs = outputs['segment_output']
            if seg_outputs.shape[-2:] != lab_0.shape[-2:]:
                seg_outputs_resize =  F.interpolate(seg_outputs, size=lab_0.shape[-2:], mode='bilinear', align_corners=False)
            else:
                seg_outputs_resize =  seg_outputs
            # Anatomical-prior-constrained hierarchical decoding (validation only)
            pred = hierarchical_decode(seg_outputs_resize, config['layer_ids'], config['fluid_ids'], config['contain_edges'])
            logits = F.one_hot(pred.long(), num_classes=config['num_classes']).permute(0, 3, 1, 2).float()
            dice_mean = dice_score(logits, lab_0, config['num_classes'], valid_classes=config['valid_classes'])
            for j, name in enumerate(class_names):
                avg_meters[name].update(dice_mean[:, j].tolist(), image_batch.size(0))

            for loss_name, loss_value in losses.items():
                meter_name = "val_loss" if loss_name == "total_loss" else f"val_{loss_name}"
                if meter_name not in avg_meters:
                    avg_meters[meter_name] = AverageMeter()
                avg_meters[meter_name].update(
                    loss_value.item() if hasattr(loss_value, "item") else loss_value)

            avg_meters['val_dice'].update(np.nanmean(dice_mean, axis=-1), image_batch.size(0))

            postfix = OrderedDict((n, m.avg) for n, m in avg_meters.items())
            loss_str = ", ".join(f"{n}: {to_float(v):.4f}" for n, v in losses.items())
            root_logger.info(f"val Losses: {loss_str}, val_dice:{postfix.get('val_dice', 0):.4f}")
            pbar.set_postfix(postfix)
            pbar.update(1)

    pbar.close()
    return postfix


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(writer, epoch, train_log, val_log, current_lr):
    writer.add_scalar('lr', current_lr, epoch)
    all_keys = set(train_log.keys()) | set(val_log.keys())
    metric_names = set()
    for key in all_keys:
        if key.startswith('train_'):
            metric_names.add(key[6:])
        elif key.startswith('val_'):
            metric_names.add(key[4:])

    for name in metric_names:
        scalars = {}
        if f'train_{name}' in train_log:
            scalars[f'train_{name}'] = to_float(train_log[f'train_{name}'])
        if f'val_{name}' in val_log:
            scalars[f'val_{name}'] = to_float(val_log[f'val_{name}'])
        if scalars:
            writer.add_scalars(name, scalars, epoch)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(config, output_path, model):
    optimizer, scheduler = build_optimizer(config, model)
    losses = build_losses(config)

    seed = config['seed']
    g = torch.Generator()
    g.manual_seed(seed)

    save_dir = output_path
    os.makedirs(save_dir, exist_ok=True)

    # --- logging ---
    log_file = os.path.join(save_dir, f"log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(fh)
    logging.info(str(config))

    print("training from scratch")

    # --- datasets ---
    Dataset = get_dataset_class(config['dataset'])
    dir_kwargs = {k: config[k] for k in ('img_dir', 'lab_dir', 'layer_lab_dir') if k in config}
    train_set = Dataset(dataset=config['dataset'],
                        data_type='train',
                        resize=[config['image_height'], config['image_width']],
                        mode='train',
                        seed=config['seed'],
                        **dir_kwargs)
    val_set = Dataset(dataset=config['dataset'],
                      data_type='val',
                      resize=[config['image_height'], config['image_width']],
                      mode='val',
                      seed=config['seed'],
                      **dir_kwargs)

    def worker_init_fn(wid):
        np.random.seed(seed + wid)
        random.seed(seed + wid)

    bs = config['batch_size'] * config['n_gpu']
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, generator=g,
                              num_workers=config['num_workers'], pin_memory=True,
                              worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True, drop_last=False)

    writer = SummaryWriter(os.path.join(save_dir, 'log'))

    best_loss = 2
    best_dice = 0
    trigger = 0

    for epoch in range(1, config['epochs'] + 1):
        print(f'Epoch [{epoch}/{config["epochs"]}]')
        train_log = run_epoch(config, train_loader, model, optimizer, losses, root_logger)
        val_log = run_validation(config, val_loader, model, losses, root_logger, epoch=epoch, save_dir=save_dir)

        scheduler.step(val_log['val_loss'])

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}: lr={current_lr}, "
                     f"val_dice={val_log['val_dice']}, "
                     f"val_loss={val_log['val_loss']}")

        log_metrics(writer, epoch, train_log, val_log, current_lr)

        # checkpointing
        if val_log['val_dice'] > best_dice:
            trigger = 0
            best_loss = min(best_loss, val_log['val_loss'])
            best_dice = val_log['val_dice']
            ckpt_dir = os.path.join(save_dir, 'checkpoint')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
            logging.info(f'saved best model at epoch {epoch}, val_loss={best_loss:.4f}')
            print(f"=> saved best model at epoch {epoch}")
        else:
            trigger += 1
            logging.info(f"No improvement for {trigger} epochs")

        logging.info(f'best_val_loss={best_loss:.4f}')

        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

    writer.close()
    del train_loader, val_loader
    gc.collect()
