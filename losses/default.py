import torch

from losses.loss_ce import CELoss
from losses.loss_dice import DiceLoss
from losses.loss_kl_bscan import AttentionLabelKLLoss as BscanAttentionKLLoss
from losses.loss_sup_ascan import AscanSupConLossFromLabel
from losses.wbceloss import weighted_bce_with_logits as wbce


# Weights for the common loss presets. Paper loss:
#   L = L_seg + 0.1 * L_ARA + 0.5 * L_AttnCon + 0.5 * L_AttnKL,
#   L_seg = CE + Dice.
LOSS_WEIGHTS = {
    "ce": {"ce": 1},
    "dice": {"dice": 1},
    "ce_dice": {"ce": 1, "dice": 1},
    "ce_dice_asup_bkl": {"ce": 1, "dice": 1, "sup_ascan": 0.5, "kl_bscan": 0.5},
    "ce_dice_wbce": {"ce": 1, "dice": 1, "wbce": 0.1},
    "ce_dice_wbce_asup_bkl": {
        "ce": 1, "dice": 1, "wbce": 0.1, "sup_ascan": 0.5, "kl_bscan": 0.5,
    },
}


def build_losses(args):
    """Instantiate the loss modules once; :func:`criterion` applies them per batch."""
    losstype = args['loss'].lower()
    weights = LOSS_WEIGHTS.get(losstype, {"ce": 1, "dice": 1})

    ignore_index = args.get('ignore_index')
    modules = {}
    for name in weights:
        if name == "ce":
            modules[name] = CELoss(args['num_classes'], args.get('class_weight'), device=args['device'])
        elif name == "dice":
            modules[name] = DiceLoss(args['num_classes'], device=args['device'])
        elif name == "kl_bscan":
            modules[name] = BscanAttentionKLLoss(remove_self=True, ignore_index=ignore_index, per_head=False)
        elif name == "sup_ascan":
            modules[name] = AscanSupConLossFromLabel(
                temperature=0.5, remove_self=True, per_head=False, ignore_index=ignore_index)
        elif name == "wbce":
            # (lesion, layer) pairs in the config -> (layer, lesion) expected by wbce.
            modules[name] = wbce(
                [(dst, src) for (src, dst) in args['contain_edges']],
                args['directed_edges'],
                args['num_classes'],
                device=args['device'])
        else:
            raise ValueError(f"Loss '{name}' is not defined.")
    return modules


def criterion(args, target, output, losses=None, loss_weight=None, iter_num=None):
    """Compute the weighted loss terms from ``target`` / ``output``.

    ``losses`` is the pre-built module dict from :func:`build_losses`; when
    omitted it is built on the fly (kept for backward compatibility).
    """
    loss_dict = {"total_loss": 0}
    if iter_num is not None:
        loss_dict['iter_num'] = int(iter_num)

    losstype = args['loss'].lower()
    weights = LOSS_WEIGHTS.get(losstype, {"ce": 1, "dice": 1})
    if losses is None:
        losses = build_losses(args)

    total_loss = 0
    for name, w in weights.items():
        if w <= 0:
            loss_dict[f"{name}_loss"] = torch.tensor(0.0, device=args['device'])
            continue
        value = losses[name].to(args['device'])(output, target)
        if value is None:
            loss_dict[f"{name}_loss"] = torch.tensor(0.0, device=args['device'])
            continue
        total_loss += w * value
        loss_dict[f"{name}_loss"] = w * value

    loss_dict["total_loss"] = total_loss
    return loss_dict
