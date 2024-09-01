import numpy as np
from PIL import Image
import pywt
import argparse
import os
import cv2
from tqdm import tqdm
from scipy.signal import correlate2d
from scipy import signal
from sklearn.metrics import mutual_info_score

def gaussian_kernel(shape, std, normalised=False):

    std = std
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='.../GobletNet/demo_image')
    parser.add_argument('--wavelet_path', default='.../GobletNet/demo_wavelet')
    parser.add_argument('--save_HFIR_overlaid_wavelet_path', default='.../GobletNet/HFIR_overlaid_wavelet')
    parser.add_argument('--save_NI_overlaid_wavelet_path', default='.../GobletNet/NI_overlaid_wavelet')
    parser.add_argument('--save_DR_overlaid_wavelet_path', default='.../GobletNet/DR_overlaid_wavelet')
    parser.add_argument('--save_HFIR_overlaid_image_path', default='.../GobletNet/figure/HFIR_overlaid_image')
    parser.add_argument('--save_NI_overlaid_image_path', default='.../GobletNet/figure/NI_overlaid_image')
    parser.add_argument('--save_DR_overlaid_image_path', default='.../GobletNet/figure/DR_overlaid_image')
    parser.add_argument('--kernel_size', default=2, type=int)
    parser.add_argument('--wavelet_heatmap_weight', default=0.5, type=float)
    parser.add_argument('--image_heatmap_weight', default=0.3, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.save_HFIR_overlaid_wavelet_path):
        os.mkdir(args.save_HFIR_overlaid_wavelet_path)
    if not os.path.exists(args.save_NI_overlaid_wavelet_path):
        os.mkdir(args.save_NI_overlaid_wavelet_path)
    if not os.path.exists(args.save_DR_overlaid_wavelet_path):
        os.mkdir(args.save_DR_overlaid_wavelet_path)

    if not os.path.exists(args.save_HFIR_overlaid_image_path):
        os.mkdir(args.save_HFIR_overlaid_image_path)
    if not os.path.exists(args.save_NI_overlaid_image_path):
        os.mkdir(args.save_NI_overlaid_image_path)
    if not os.path.exists(args.save_DR_overlaid_image_path):
        os.mkdir(args.save_DR_overlaid_image_path)

    for i in tqdm(os.listdir(args.image_path)):

        image_path = os.path.join(args.image_path, i)
        wavelet_path = os.path.join(args.wavelet_path, i)

        save_HFIR_overlaid_wavelet_path = os.path.join(args.save_HFIR_overlaid_wavelet_path, i)
        save_NI_overlaid_wavelet_path = os.path.join(args.save_NI_overlaid_wavelet_path, i)
        save_DR_overlaid_wavelet_path = os.path.join(args.save_DR_overlaid_wavelet_path, i)
        save_HFIR_overlaid_image_path = os.path.join(args.save_HFIR_overlaid_image_path, i)
        save_NI_overlaid_image_path = os.path.join(args.save_NI_overlaid_image_path, i)
        save_DR_overlaid_image_path = os.path.join(args.save_DR_overlaid_image_path, i)

        wavelet = Image.open(wavelet_path)
        wavelet_array = np.array(wavelet)

        image = Image.open(image_path).resize([wavelet_array.shape[1], wavelet_array.shape[0]])
        image_array = np.array(image)
        assert image_array.shape[0] == wavelet_array.shape[0]
        assert image_array.shape[1] == wavelet_array.shape[1]

        if len(image_array.shape) == 2:
            image_array = np.stack((image_array, image_array, image_array), axis=2)

        h, w = wavelet_array.shape
        HFIR_heatmap = np.zeros(wavelet_array.shape, dtype=np.float64)
        NI_heatmap = np.zeros(wavelet_array.shape, dtype=np.float64)

        for row in range(h):
            for col in range(w):
                up_y = np.max([0, row - args.kernel_size])
                down_y = np.min([h, row + args.kernel_size+1])
                left_x = np.max([0, col - args.kernel_size])
                right_x = np.min([w, col + args.kernel_size+1])
                region = wavelet_array[up_y:down_y, left_x:right_x]
                HFIR_heatmap[row, col], NI_heatmap[row, col] = entropy(region)

        HFIR_heatmap = (HFIR_heatmap - np.min(HFIR_heatmap)) / (np.max(HFIR_heatmap) - np.min(HFIR_heatmap))
        NI_heatmap = (NI_heatmap - np.min(NI_heatmap)) / (np.max(NI_heatmap) - np.min(NI_heatmap))

        DR_heatmap = HFIR_heatmap - NI_heatmap
        DR_heatmap[DR_heatmap <= 0] = 0
        DR_heatmap = (DR_heatmap - np.min(DR_heatmap)) / (np.max(DR_heatmap) - np.min(DR_heatmap))

        HFIR_heatmap = cv2.applyColorMap(np.uint8(HFIR_heatmap * 255), cv2.COLORMAP_PARULA)
        HFIR_heatmap = cv2.cvtColor(HFIR_heatmap, cv2.COLOR_BGR2RGB)
        HFIR_heatmap = np.float32(HFIR_heatmap) / 255

        NI_heatmap = cv2.applyColorMap(np.uint8(NI_heatmap * 255), cv2.COLORMAP_PARULA)
        NI_heatmap = cv2.cvtColor(NI_heatmap, cv2.COLOR_BGR2RGB)
        NI_heatmap = np.float32(NI_heatmap) / 255

        DR_heatmap = cv2.applyColorMap(np.uint8(DR_heatmap * 255), cv2.COLORMAP_PARULA)
        DR_heatmap = cv2.cvtColor(DR_heatmap, cv2.COLOR_BGR2RGB)
        DR_heatmap = np.float32(DR_heatmap) / 255

        # show on wavelet
        wavelet_array = (wavelet_array - np.min(wavelet_array)) / (wavelet_array.max() - wavelet_array.min())
        wavelet_array = np.float32(np.stack((wavelet_array, wavelet_array, wavelet_array), axis=2))

        HFIR_wavelet = args.wavelet_heatmap_weight * HFIR_heatmap + (1 - args.wavelet_heatmap_weight) * wavelet_array
        HFIR_wavelet = 255 * (HFIR_wavelet - np.min(HFIR_wavelet)) / (np.max(HFIR_wavelet) - np.min(HFIR_wavelet))
        HFIR_wavelet = Image.fromarray(HFIR_wavelet.astype(np.uint8))
        HFIR_wavelet.save(save_HFIR_overlaid_wavelet_path)

        NI_wavelet = args.wavelet_heatmap_weight * NI_heatmap + (1 - args.wavelet_heatmap_weight) * wavelet_array
        NI_wavelet = 255 * (NI_wavelet - np.min(NI_wavelet)) / (np.max(NI_wavelet) - np.min(NI_wavelet))
        NI_wavelet = Image.fromarray(NI_wavelet.astype(np.uint8))
        NI_wavelet.save(save_NI_overlaid_wavelet_path)

        DR_wavelet = args.wavelet_heatmap_weight * DR_heatmap + (1 - args.wavelet_heatmap_weight) * wavelet_array
        DR_wavelet = 255 * (DR_wavelet - np.min(DR_wavelet)) / (np.max(DR_wavelet) - np.min(DR_wavelet))
        DR_wavelet = Image.fromarray(DR_wavelet.astype(np.uint8))
        DR_wavelet.save(save_DR_overlaid_wavelet_path)

        # show on image
        image_array = np.float32(image_array) / 255

        HFIR_image = args.image_heatmap_weight * HFIR_heatmap + (1 - args.image_heatmap_weight) * image_array
        HFIR_image = 255 * (HFIR_image - np.min(HFIR_image)) / (np.max(HFIR_image) - np.min(HFIR_image))
        HFIR_image = Image.fromarray(HFIR_image.astype(np.uint8))
        HFIR_image.save(save_HFIR_overlaid_image_path)

        NI_image = args.image_heatmap_weight * NI_heatmap + (1 - args.image_heatmap_weight) * image_array
        NI_image = 255 * (NI_image - np.min(NI_image)) / (np.max(NI_image) - np.min(NI_image))
        NI_image = Image.fromarray(NI_image.astype(np.uint8))
        NI_image.save(save_NI_overlaid_image_path)

        DR_image = args.image_heatmap_weight * DR_heatmap + (1 - args.image_heatmap_weight) * image_array
        DR_image = 255 * (DR_image - np.min(DR_image)) / (np.max(DR_image) - np.min(DR_image))
        DR_image = Image.fromarray(DR_image.astype(np.uint8))
        DR_image.save(save_DR_overlaid_image_path)