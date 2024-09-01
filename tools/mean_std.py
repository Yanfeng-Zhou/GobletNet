import os
import cv2
import numpy as np
import argparse
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_dir', default='.../GobletNet/dataset/CREMI/train/HH')
    parser.add_argument('--max_pixel', default=255.0)
    parser.add_argument('--if_RGB', default=False)
    args = parser.parse_args()

    # Calculate  mean
    if args.if_RGB:
        channel_1 = 0
        channel_2 = 0
        channel_3 = 0
    else:
        channel_1 = 0

    count = 0
    pixels_num = 0

    for image in os.listdir(args.data_dir):

        image_path = os.path.join(args.data_dir, image)
        image = Image.open(image_path)
        img = np.array(image) / args.max_pixel
        count += 1

        if args.if_RGB:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        pixels_num += h*w

        if args.if_RGB:
            channel_1 = channel_1 + np.sum(img[:, :, 0])
            channel_2 = channel_2 + np.sum(img[:, :, 1])
            channel_3 = channel_3 + np.sum(img[:, :, 2])
        else:
            channel_1 = channel_1 + np.sum(img[:, :])

    print('number of images: {}'.format(count))
    print('number of pixels: {}'.format(pixels_num))

    if args.if_RGB:
        mean_1 = channel_1 / pixels_num
        mean_2 = channel_2 / pixels_num
        mean_3 = channel_3 / pixels_num
        print("mean_1 is %f, mean_2 is %f, mean_3 is %f" % (mean_1, mean_2, mean_3))
    else:
        mean_1 = channel_1 / pixels_num
        print("mean_1 is %f" % (mean_1))

    # Calculate std
    if args.if_RGB:
        channel_1 = 0
        channel_2 = 0
        channel_3 = 0
    else:
        channel_1 = 0

    count = 0
    pixels_num = 0

    for image in os.listdir(args.data_dir):

        image_path = os.path.join(args.data_dir, image)
        image = Image.open(image_path)
        img = np.array(image) / args.max_pixel
        count += 1

        if args.if_RGB:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        pixels_num += h * w

        if args.if_RGB:
            channel_1 = channel_1 + np.sum((img[:, :, 0] - mean_1)**2)
            channel_2 = channel_2 + np.sum((img[:, :, 1] - mean_2)**2)
            channel_3 = channel_3 + np.sum((img[:, :, 2] - mean_3)**2)
        else:
            channel_1 = channel_1 + np.sum((img[:, :] - mean_1)**2)

    print('number of images: {}'.format(count))
    print('number of pixels: {}'.format(pixels_num))

    if args.if_RGB:
        std_1 = np.sqrt(channel_1 / pixels_num)
        std_2 = np.sqrt(channel_2 / pixels_num)
        std_3 = np.sqrt(channel_3 / pixels_num)
        print("std_1 is %f, std_2 is %f, std_3 is %f" % (std_1, std_2, std_3))
    else:
        std_1 = np.sqrt(channel_1 / pixels_num)
        print("std_1 is %f" % (std_1))
