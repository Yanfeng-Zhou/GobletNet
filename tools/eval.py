from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os
from PIL import Image
from medpy.metric.binary import hd95, assd
import albumentations as A
import SimpleITK as sitk
from config.dataset_config.dataset_cfg import dataset_cfg


def eval_distance(mask_list, seg_result_list, num_classes):

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    assert len(mask_list) == len(seg_result_list)
    if num_classes == 2:
        hd_list = []
        sd_list = []
        for i in range(len(mask_list)):

            if np.any(seg_result_list[i]) and np.any(mask_list[i]):

                hd_ = hd95(seg_result_list[i], mask_list[i])
                sd_ = assd(seg_result_list[i], mask_list[i])
                hd_list.append(hd_)
                sd_list.append(sd_)

        hd = np.mean(hd_list)
        sd = np.mean(sd_list)

        print('| Hd: {:.4f}'.format(hd).ljust(print_num_minus, ' '), '|')
        print('| Sd: {:.4f}'.format(sd).ljust(print_num_minus, ' '), '|')

    else:
        hd_list = []
        sd_list = []

        for cls in range(num_classes-1):

            hd_list_ = []
            sd_list_ = []

            for i in range(len(mask_list)):

                mask_list_ = mask_list[i].copy()
                seg_result_list_ = seg_result_list[i].copy()

                mask_list_[mask_list[i] != (cls + 1)] = 0
                seg_result_list_[seg_result_list[i] != (cls + 1)] = 0

                if np.any(seg_result_list_) and np.any(mask_list_):
                    hd_ = hd95(seg_result_list_, mask_list_)
                    sd_ = assd(seg_result_list_, mask_list_)
                    hd_list_.append(hd_)
                    sd_list_.append(sd_)

            hd = np.mean(hd_list_)
            sd = np.mean(sd_list_)

            hd_list.append(hd)
            sd_list.append(sd)

        hd_list = np.array(hd_list)
        sd_list = np.array(sd_list)

        m_hd = np.mean(hd_list)
        m_sd = np.mean(sd_list)

        np.set_printoptions(precision=4, suppress=True)
        print('|  Hd: {}'.format(hd_list).ljust(print_num_minus, ' '), '|')
        print('|  Sd: {}'.format(sd_list).ljust(print_num_minus, ' '), '|')
        print('| mHd: {:.4f}'.format(m_hd).ljust(print_num_minus, ' '), '|')
        print('| mSd: {:.4f}'.format(m_sd).ljust(print_num_minus, ' '), '|')

    print('-' * print_num)

def eval_pixel(mask_list, seg_result_list, num_classes):

    c = confusion_matrix(mask_list, seg_result_list)

    hist_diag = np.diag(c)
    hist_sum_0 = c.sum(axis=0)
    hist_sum_1 = c.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)

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
    parser.add_argument('--pred_path', default='.../GobletNet/seg_pred/CREMI/fusionnet-l=0.1-e=200-s=50-g=0.5-b=4-w=20')
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

        pred_list.append(pred)
        mask_list.append(mask)

        if num == 0:
            pred_flatten_list = pred.flatten()
            mask_flatten_list = mask.flatten()
        else:
            pred_flatten_list = np.append(pred_flatten_list, pred.flatten())
            mask_flatten_list = np.append(mask_flatten_list, mask.flatten())

        num += 1

    eval_pixel(mask_flatten_list, pred_flatten_list, cfg['NUM_CLASSES'])
    eval_distance(mask_list, pred_list, cfg['NUM_CLASSES'])

