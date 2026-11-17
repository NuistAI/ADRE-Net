import os
import numpy as np
import cv2

from configs.color_set import get_color_set


def vis_result(img_ids, img_datas, label, preds, save_dir, config,
               label_sdfs=None, preds_sdfs=None, draw_diff=False):
    """Visualize and save segmentation results for a batch of images."""
    dataset_name = config['dataset']
    color_set = get_color_set(dataset_name)
    n_class = config['num_classes']

    if img_datas.dim() == 3:
        img_datas = img_datas.unsqueeze(1)
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if label.dim() == 2:
        label = label.unsqueeze(1)
    img_datas = img_datas.data.cpu().numpy().astype('uint8')

    pred_onehot = preds.argmax(1).clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu()
    batch_size = img_datas.shape[0]

    for i in range(batch_size):
        img = img_datas[i]
        if img.max() <= 2:
            img = img * 256
        imn_name = img_ids[i]
        img = np.squeeze(img)
        img = gray2rgbimage(img)

        pred = preds[i]
        prediction_label = pred_onehot[i]
        pred_img = draw_mask( prediction_label, color_set, n_class=n_class)

        if label is not None:
            ground_truth = label.data.cpu().numpy()[i]
            gt_img = draw_mask(ground_truth, color_set, n_class=n_class)
            incorrect_img = mark_incorrect_pixels( prediction_label, ground_truth,
                                                  color_set, n_class=n_class)

            line_width = 5
            line = np.zeros((line_width, img.shape[1], img.shape[2]), dtype=np.uint8)
            show_img_1 = np.vstack((img, line, incorrect_img)).astype('uint8')
            show_img_2 = np.vstack((gt_img, line, pred_img))
            line_w = np.zeros((show_img_1.shape[0], line_width, show_img_1.shape[2]), dtype=np.uint8)
            combined_img = np.hstack((show_img_1, line_w, show_img_2))
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, imn_name + '.png'), combined_img)
        else:
            cv2.imwrite(os.path.join(save_dir, imn_name),
                        np.hstack((img, pred_img)).astype('uint8'))


def mark_incorrect_pixels(prediction, ground_truth, color_set, n_class):
    assert prediction.shape == ground_truth.shape, "Prediction and ground truth shapes mismatch"
    mask = np.zeros_like(prediction)
    incorrect_indices = np.where(prediction != ground_truth)
    mask[incorrect_indices] = 1
    incorrect_gt = np.multiply(mask, ground_truth)
    incorrect_img = draw_mask(incorrect_gt, color_set, n_class=n_class)
    return incorrect_img


def draw_img(imgs, segs, color_set, title=None, n_class=8):
    label_set = [i for i in range(n_class)]
    mask = np.zeros_like(imgs)
    if title is not None:
        mask = cv2.putText(mask, title, (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
    for draw_label in label_set:
        mask[:, :, 0][segs[:, :] == draw_label] = color_set[draw_label][0]
        mask[:, :, 1][segs[:, :] == draw_label] = color_set[draw_label][1]
        mask[:, :, 2][segs[:, :] == draw_label] = color_set[draw_label][2]
    mask = mask.astype('uint8')
    img_mask = cv2.addWeighted(imgs, 0.4, mask, 0.6, 0)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    return img_mask

def draw_mask(segs, color_set, n_class=8):
    h, w = segs.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for draw_label in range(n_class):
        mask[segs == draw_label] = color_set[draw_label]
    # OpenCV defaults to BGR; convert to RGB before returning
    mask = cv2.cvtColor(mask,  cv2.COLOR_RGB2BGR)

    return mask

def gray2rgbimage(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype('uint8')
    return image.astype('uint8')
