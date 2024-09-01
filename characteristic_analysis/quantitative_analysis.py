import tqdm_pathos
import numpy as np
from PIL import Image
import argparse
import os
from scipy.signal import correlate2d
from scipy import signal

def gaussian_kernel(shape, std, normalised=False):

    gaussian1D_row = signal.gaussian(shape[0], std)
    gaussian1D_col = signal.gaussian(shape[1], std)
    gaussian2D = np.outer(gaussian1D_row, gaussian1D_col)

    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def entropy(image):

    mean = image.sum()
    std = image.std()
    if std == 0:
        return std, mean / (image.shape[0] * image.shape[1])
    else:
        gaussian = gaussian_kernel(image.shape, std, True)
        correlation = correlate2d(image, gaussian, mode='same', boundary='symm')
        return std, correlation.mean()

def quantitative_analysis(wavelet_list, wavelet_path, kernel_size):
    wavelet = np.array(Image.open(os.path.join(wavelet_path, wavelet_list)))

    h, w = wavelet.shape
    HF_heatmap = np.zeros(wavelet.shape, dtype=np.float64)
    noise_heatmap = np.zeros(wavelet.shape, dtype=np.float64)

    for row in range(h):
        for col in range(w):
            up_y = np.max([0, row - kernel_size])
            down_y = np.min([h, row + kernel_size + 1])
            left_x = np.max([0, col - kernel_size])
            right_x = np.min([w, col + kernel_size + 1])
            region = wavelet[up_y:down_y, left_x:right_x]
            HF_heatmap[row, col], noise_heatmap[row, col] = entropy(region)

    HF_heatmap = (HF_heatmap - np.min(HF_heatmap)) / (np.max(HF_heatmap) - np.min(HF_heatmap))
    noise_heatmap = (noise_heatmap - np.min(noise_heatmap)) / (np.max(noise_heatmap) - np.min(noise_heatmap))

    contour_heatmap = HF_heatmap - noise_heatmap
    contour_heatmap[contour_heatmap <= 0] = 0
    contour_heatmap = (contour_heatmap - np.min(contour_heatmap)) / (np.max(contour_heatmap) - np.min(contour_heatmap))

    return HF_heatmap.mean(), HF_heatmap.mean() - contour_heatmap.mean(), contour_heatmap.mean()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelet_path', default='.../dataset/Cityscapes/train/leftImg8bit')
    parser.add_argument('--kernel_size', default=2, type=int)
    args = parser.parse_args()

    result = tqdm_pathos.map(quantitative_analysis, os.listdir(args.wavelet_path), args.wavelet_path, args.kernel_size)

    HF_heatmap_list = []
    noise_heatmap_list = []
    contour_heatmap_list = []
    for i in result:
        HF_heatmap_list.append(i[0])
        noise_heatmap_list.append(i[1])
        contour_heatmap_list.append(i[2])
    print('HF_heatmap:', np.array(HF_heatmap_list).mean(), np.array(HF_heatmap_list).std())
    print('noise_heatmap:', np.array(noise_heatmap_list).mean(), np.array(noise_heatmap_list).std())
    print('contour_heatmap', np.array(contour_heatmap_list).mean(), np.array(contour_heatmap_list).std())