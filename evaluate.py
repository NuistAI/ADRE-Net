import argparse
import csv
import gc
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from adre_net import choice_backbone
from dataset import get_dataset_class
from visualize import vis_result
from metrics import dice_score, AverageMeter
from anatomy_metrics import (mad_metrics, fragmentation_count, layer_cross_stats, fluid_metrics, hierarchical_decode)


def evaluate(config, data_loader, model, class_name,
             results_dir, dataset_type='test', pre_model_path='model.pth',
             is_draw=False):
    avg_meters = {'dice': AverageMeter()}
    # Anatomical prior: layer/fluid ids and MAD boundaries, from config
    layer_ids = config['layer_ids']        # top/bottom background + 3 layers, fluids excluded
    fluid_ids = config['fluid_ids']        # PED / SRF / IRF
    # MAD boundaries: interfaces between adjacent layers; fluids report no boundary
    mad_boundary_spec = [(c, 'top') for c in layer_ids[1:]]
    mad_boundary_names = [f'mad_b{layer_ids[i]}_{layer_ids[i+1]}' for i in range(len(layer_ids) - 1)]
    new_metric_rows = []
    new_metric_index = []
    for i in range(len(class_name)):
        avg_meters[class_name[i]] = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader))
        for _, sampled_batch in enumerate(data_loader):
            image_batch = sampled_batch['image'].to(config['device'])
            lab_0 = sampled_batch['lab_0'].to(config['device'])
            label_batch = sampled_batch['label'].to(config['device'])
            inputs = {'image': image_batch, 'label': label_batch}
            outputs = model(inputs)
            seg_outputs = outputs['segment_output']
            image_0 = sampled_batch['image_0']

            if seg_outputs.shape[-2:] != lab_0.shape[-2:]:
                seg_outputs_resize =  F.interpolate(seg_outputs, size=lab_0.shape[-2:], mode='bilinear', align_corners=False)
            else:
                seg_outputs_resize =  seg_outputs

            # Anatomical-prior-constrained hierarchical decoding (inference only)
            pred_resize = hierarchical_decode(seg_outputs_resize, layer_ids, fluid_ids, config['contain_edges'])
            logits_resize = F.one_hot(pred_resize.long(), num_classes=config['num_classes']).permute(0, 3, 1, 2).float()
            pred_full = hierarchical_decode(seg_outputs, layer_ids, fluid_ids, config['contain_edges'])
            logits_full = F.one_hot(pred_full.long(), num_classes=config['num_classes']).permute(0, 3, 1, 2).float()

            if is_draw:
                    save_img_path = os.path.join(results_dir, 'Results visualization',
                                                    f'{dataset_type}_{pre_model_path}')
                    os.makedirs(save_img_path, exist_ok=True)
                    vis_result(sampled_batch['img_id'], image_0,
                                lab_0, logits_resize, save_img_path, config)

            score = dice_score(logits_resize, lab_0, config['num_classes'],
                               valid_classes=config['valid_classes'])

            # MAD boundary distance / layer fragmentation count / layer-order violation count
            mad_per_boundary, _ = mad_metrics(
                logits_resize, lab_0, config['num_classes'],
                boundary_spec=mad_boundary_spec,
                pixel_spacing=config['pixel_spacing'])
            frag_count, _ = fragmentation_count(logits_full, layer_ids, fluid_ids)   # [B]
            _, nviol, _ = layer_cross_stats(logits_full, layer_ids)  # [B]
            if fluid_ids:
                fluid_precision, fluid_recall = fluid_metrics(
                    logits_resize, lab_0, fluid_ids)     # fluid classes merged into one class [B]

            # Record metrics per sample
            for b, img_id in enumerate(sampled_batch['img_id']):
                new_row = {name: mad_per_boundary[b, k] for k, name in enumerate(mad_boundary_names)}
                row_update = {
                    'FragCount': frag_count[b],
                    'Nviol': nviol[b],
                }
                if fluid_ids:
                    row_update.update({
                        'Fluid_Precision': fluid_precision[b],
                        'Fluid_Recall': fluid_recall[b],
                    })
                new_row.update(row_update)
                new_metric_rows.append(new_row)
                new_metric_index.append(img_id)

            for j in range(len(class_name)):
                avg_meters[class_name[j]].update(score[:, j].tolist(), image_batch.size(0))

            avg_meters['dice'].update(np.nanmean(score, axis=1), image_batch.size(0))
            if 'best' in pre_model_path:
                    img_ids = sampled_batch['img_id']

                    # Per-image Dice (mean + per-class)
                    img_scores = pd.DataFrame({'dice': np.nanmean(score, axis=1)}, index=img_ids)
                    for j, name in enumerate(class_name):
                        img_scores[name] = score[:, j]

                    file_path = os.path.join(results_dir, 'per_image_dice.csv')
                    fieldnames = ['Index'] + list(img_scores.columns)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for index, row in img_scores.iterrows():
                            writer.writerow([index] + row.tolist())

            postfix = OrderedDict([('dice', 0.0)])
            dice_sum = []
            for name in class_name:
                postfix[name] = round(avg_meters[name].avg * 100, 2)
                dice_sum.append(postfix[name])
            avgdice = sum(dice_sum) / len(class_name)
            postfix['dice'] = round(avgdice, 2)

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    # Per-sample CSV for new metrics
    new_metric_df = pd.DataFrame(new_metric_rows, index=new_metric_index)
    new_metric_df.index.name = 'img_id'
    new_metric_path = os.path.join(results_dir, 'per_sample_metrics.csv')
    new_metric_df.to_csv(new_metric_path)

    # Aggregate: MAD per boundary + LFI/LFS (layer fragmentation) + LOVI/LOVS (layer-order violation) + fluid
    mad_per_boundary_mean = new_metric_df[mad_boundary_names].mean()      # mean MAD per boundary (um)

    frag_count = new_metric_df['FragCount']
    lfi = (frag_count > 0).mean() * 100          # layer-fragmentation incidence (%)
    lfs = frag_count[frag_count > 0].mean() if (frag_count > 0).any() else 0.0   # severity (mean over positive images)

    nviol = new_metric_df['Nviol']
    lovi = (nviol > 0).mean() * 100              # layer-order violation incidence (%)
    lovs = nviol[nviol > 0].mean() if (nviol > 0).any() else 0.0                  # severity (mean over positive images)

    for name in mad_boundary_names:
        postfix[name] = round(float(mad_per_boundary_mean[name]), 4)
    postfix['mad'] = round(float(mad_per_boundary_mean.mean()), 4)
    postfix['lfi'] = round(float(lfi), 4)
    postfix['lfs'] = round(float(lfs), 4)
    postfix['lovi'] = round(float(lovi), 4)
    postfix['lovs'] = round(float(lovs), 4)
    if fluid_ids:
        fluid_precision_mean = new_metric_df['Fluid_Precision'].mean() * 100
        fluid_recall_mean = new_metric_df['Fluid_Recall'].mean() * 100
        postfix['fluid_precision'] = round(float(fluid_precision_mean), 4)
        postfix['fluid_recall'] = round(float(fluid_recall_mean), 4)

    return postfix


def get_class_name(config):
    return config['class_names']


def main_test(config, results_dir, model, dataset_type='test',
              pre_model_path='model.pth', is_draw=False):
    output_results_dir = results_dir
    os.makedirs(output_results_dir, exist_ok=True)

    if 'best' in pre_model_path:
        model_path = os.path.join(output_results_dir, 'checkpoint', pre_model_path)
    else:
        model_path = os.path.join(output_results_dir, pre_model_path)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.cuda()
    model.eval()

    Dataset = get_dataset_class(config['dataset'])
    dir_kwargs = {k: config[k] for k in ('img_dir', 'lab_dir', 'layer_lab_dir') if k in config}
    dataset = Dataset(
        dataset=config['dataset'],
        data_type=dataset_type,
        resize=[config['image_height'], config['image_width']],
        mode='test',
        seed=config['seed'],
        **dir_kwargs,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    class_name = get_class_name(config)
    postfix = evaluate(config, data_loader, model, class_name,
                       results_dir, dataset_type, pre_model_path=pre_model_path,
                       is_draw=is_draw)

    del data_loader
    gc.collect()
    time.sleep(0.5)
    return postfix


def get_config_from_path(path):
    config_path = os.path.join(path, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', help='path to model checkpoint directory')
    args = parser.parse_args()

    results_dir = args.checkpoint
    config = get_config_from_path(results_dir)

    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    model = choice_backbone(config['backbone'], config['num_classes'], input_config=config)
    main_test(config, results_dir, model, is_draw=True)
