from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os
from PIL import Image
from medpy.metric.binary import hd95, assd
import albumentations as A
import SimpleITK as sitk
from config.dataset_config.dataset_cfg import dataset_cfg
from skimage import measure, morphology


def eval_pixel(mask_list, seg_result_list, num_classes):

    c = confusion_matrix(mask_list, seg_result_list)

    hist_diag = np.diag(c)
    hist_sum_0 = c.sum(axis=0)
    hist_sum_1 = c.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)

    print(hist_diag)

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    print('-' * print_num)
    if num_classes > 2:
        m_jaccard = np.nanmean(jaccard)
        m_dice = np.nanmean(dice)
        np.set_printoptions(precision=4, suppress=True)
        print('|  Jc: {}'.format(jaccard).ljust(print_num_minus, ' '), '|')
        print('|  Dc: {}'.format(dice).ljust(print_num_minus, ' '), '|')
        print('| mJc: {:.4f}'.format(m_jaccard).ljust(print_num_minus, ' '), '|')
        print('| mDc: {:.4f}'.format(m_dice).ljust(print_num_minus, ' '), '|')
    else:
        print('| Jc: {:.4f}'.format(jaccard[1]).ljust(print_num_minus, ' '), '|')
        print('| Dc: {:.4f}'.format(dice[1]).ljust(print_num_minus, ' '), '|')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='.../GobletNet/seg_pred/CREMI/unet-l=0.1-e=200-s=50-g=0.5-b=4-w=20')
    parser.add_argument('--mask_path', default='.../GobletNet/dataset/CREMI/val/mask')
    parser.add_argument('--dataset_name', default='CREMI', help='EPFL, CREMI, SNEMI3D, UroCell, MitoEM, Nanowire, BetaSeg')
    args = parser.parse_args()

    cfg = dataset_cfg(args.dataset_name)

    pred_list = []
    mask_list = []

    pred_flatten_list = []
    mask_flatten_list = []

    num = 0

    for i in os.listdir(args.mask_path):
        pred_path = os.path.join(args.pred_path, i)
        mask_path = os.path.join(args.mask_path, i)

        pred = Image.open(pred_path)
        pred = np.array(pred)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        resize = A.Resize(cfg['SIZE'][1], cfg['SIZE'][0], p=1)(image=pred, mask=mask)
        mask = resize['mask']
        pred = resize['image']

        if cfg['NUM_CLASSES'] == 2:

            pred_contour = pred - morphology.binary_erosion(pred)
            mask_contour = mask - morphology.binary_erosion(mask)

        else:

            pred_contour = np.zeros(pred.shape)
            mask_contour = np.zeros(mask.shape)

            for k in range(cfg['NUM_CLASSES']):
                pred_ = pred.copy()
                pred_[pred != (k + 1)] = 0
                pred_[pred == (k + 1)] = 1
                pred_contour_class = pred_ - morphology.binary_erosion(pred_)
                pred_contour[pred_contour_class == 1] = k + 1

                mask_ = mask.copy()
                mask_[mask != (k + 1)] = 0
                mask_[mask == (k + 1)] = 1
                mask_contour_class = mask_ - morphology.binary_erosion(mask_)
                mask_contour[mask_contour_class == 1] = k + 1

        pred_list.append(pred_contour)
        mask_list.append(mask_contour)

        if num == 0:
            pred_flatten_list = pred_contour.flatten()
            mask_flatten_list = mask_contour.flatten()
        else:
            pred_flatten_list = np.append(pred_flatten_list, pred_contour.flatten())
            mask_flatten_list = np.append(mask_flatten_list, mask_contour.flatten())

        num += 1

    eval_pixel(mask_flatten_list, pred_flatten_list, cfg['NUM_CLASSES'])

