from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import cv2


def crop_single_instance(img, centroid, target_length=64):
    """
    :param img: original img
    :param centroid: centroid of the instance
    :param target_length: target size of the cropped instance
    :return: cropped instance
    """
    height, width = img.shape[:2]

    start_x, start_y, end_x, end_y = [centroid[1]-target_length//2, centroid[0]-target_length//2,
                      centroid[1]+target_length//2, centroid[0]+target_length//2]

    delta_height, delta_width = end_y - start_y, end_x - start_x
    res = np.zeros((delta_height, delta_width, img.shape[2]))

    if start_x < 0:
        start_x, dsx = 0, -start_x
        # padding left
        return None
    else:
        dsx = 0

    if end_x > width:
        end_x, dex = width, delta_width - (end_x - width)
        # padding right
        return None
    else:
        dex = delta_width

    if start_y < 0:
        start_y, dsy = 0, -start_y
        # padding up
        return None
    else:
        dsy = 0

    if end_y > height:
        end_y, dey = height, delta_height - (end_y - height)
        # padding down
        return None
    else:
        dey = delta_height

    res[dsy:dey, dsx:dex, :] = img[start_y:end_y, start_x:end_x, :]

    return res


def crop_single_instance_padding(img, centroid, target_length=64):
    """
    :param img: original img
    :param centroid: centroid of the instance
    :param target_length: target size of the cropped instance
    :return: cropped instance
    """
    # print("img.shape: ", img.shape)
    # print("centroid: ", centroid)
    height, width = img.shape[:2]

    start_x, start_y, end_x, end_y = [centroid[1]-target_length//2, centroid[0]-target_length//2,
                      centroid[1]+target_length//2, centroid[0]+target_length//2]

    delta_height, delta_width = end_y - start_y, end_x - start_x
    res = np.zeros((delta_height, delta_width, img.shape[2]))
    padding_bool = [False, False, False, False]

    if start_x < 0:
        start_x, dsx = 0, -start_x
        # padding left
        padding_bool[0] = True
    else:
        dsx = 0

    if end_x > width:
        end_x, dex = width, delta_width - (end_x - width)
        # padding right
        padding_bool[1] = True
    else:
        dex = delta_width

    if start_y < 0:
        start_y, dsy = 0, -start_y
        # padding up
        padding_bool[2] = True
    else:
        dsy = 0

    if end_y > height:
        end_y, dey = height, delta_height - (end_y - height)
        # padding down
        padding_bool[3] = True
    else:
        dey = delta_height

    res[dsy:dey, dsx:dex, :] = img[start_y:end_y, start_x:end_x, :]

    if padding_bool[0]:
        res[:, :dsx, :] = np.flip(res[:, dsx:dsx * 2, :], axis=1)

    if padding_bool[1]:
        res[:, dex:, :] = np.flip(res[:, dex - (delta_width - dex):dex, :], 1)

    if padding_bool[2]:
        res[:dsy, :, :] = np.flip(res[dsy:dsy * 2, :, :], 0)

    if padding_bool[3]:
        res[dey:, :, :] = np.flip(res[dey - (delta_height - dey):dey, :, :], 0)

    return res


def crop_img_instance_with_bg(img, mask, target_width=64, target_label=None, padding=True):
    """
    :param img: original img
    :param mask: mask of the instance
    :param target_width: target size of the cropped patch
    :return: cropped patch
    """

    crop_method = crop_single_instance_padding if padding else crop_single_instance
    if target_label is not None:
        indexes = np.where(mask == target_label)
    else:
        indexes = np.where(mask > 0.99)

    if len(indexes[0]) < 3:
        return None

    x_min, x_max = np.min(indexes[0]), np.max(indexes[0])
    y_min, y_max = np.min(indexes[1]), np.max(indexes[1])

    centroid = [(x_max + x_min) // 2, (y_max + y_min) // 2]
    if x_max == x_min or y_max == y_min:
        return None

    if (x_max - x_min < target_width) and (y_max - y_min < target_width):
        instance_now = crop_method(img, centroid, target_length=target_width)
        return instance_now

    max_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
    instance_now = crop_method(img, centroid, target_length=max_length)
    if instance_now is not None:
        instance_now = cv2.resize(instance_now, (target_width, target_width), \
                             interpolation=cv2.INTER_AREA)
    return instance_now

