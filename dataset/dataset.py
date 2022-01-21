"""
This file contains different dataset loaders.

LinearEvalMultiFCDatasetCrop is the class for multiple classification datasets, used in
sparse training;

LinearEvalSingleDatasetCrop is the class for single dataset baseline, used in
label correlation;

LinearEvalMultiFCSupUnsupDatasetCrop is the class for both classification datasets and datasets
with no labels, used in label completion.

InferenceDatasetCrop is the class for inference, loading one image and its mask
"""

from __future__ import print_function, division
import os
import pickle
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


from PIL import Image
from model.utils import dataset_mask, crop_img_instance_with_bg


def unpickle(file):
    """
    :param file: file name to read
    :return: loaded data
    """
    with open(file, 'rb') as f_in:
        dict_pickle = pickle.load(f_in, encoding='bytes')
    return dict_pickle


def get_mask(mask_path, unique_index):
    """
    :param mask_path: mask path to load mask
    :param unique_index: the unique instance index to use
    :return: mask for the instance
    """
    whole_mask = np.load(mask_path)

    pos_mask_indexes = np.where(whole_mask == unique_index)
    mask = np.zeros_like(whole_mask)
    mask[pos_mask_indexes] = 1.

    return mask


def get_mask_from_whole_mask(whole_mask, unique_index):
    """
    :param whole_mask: whole mask of all the instances in the image
    :param unique_index: the instance index for the specific instance we want to classify
    :return: the mask for the instance
    """
    pos_mask_indexes = np.where(whole_mask == unique_index)
    mask = np.zeros_like(whole_mask)
    mask[pos_mask_indexes] = 1.

    return mask


class InferenceDatasetCrop(Dataset):
    """local dataset of 64*64"""

    def __init__(self, root_dir, img_name, image_size=224):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        mask_path = os.path.join(root_dir, img_name[:-4]+"_mask.npy")

        self.mask = np.load(mask_path)
        self.instances = sorted(np.unique(self.mask))[1:]
        color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
        trans_aug = nn.Sequential(transforms.Resize(image_size))

        self.trans_aug = trans_aug

        img_path = os.path.join(root_dir, img_name)
        img = pil_loader(img_path)
        img = self.transform(img)
        self.image = color_aug(img)

    def __len__(self):
        return len(self.instances)

    def get_instances(self):
        """
        :return: the instances of the image
        """
        return self.instances

    def __getitem__(self, idx):
        unique_index = self.instances[idx]

        mask = get_mask_from_whole_mask(self.mask, unique_index)
        mask = self.transform(mask)

        instance_now = crop_img_instance_with_bg(self.image, mask)
        image = self.trans_aug(instance_now)

        return image, unique_index


class LinearEvalMultiFCDatasetCrop(Dataset):
    """local dataset of 64*64, using multiple classification datasets"""

    def __init__(self, root_dir, dataset_names, image_size=224, split_name="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_names = dataset_names

        list_paths = []
        img_instance_types = []
        for dataset_name in dataset_names:
            dataset_folder_base = os.path.join(root_dir, "40x", dataset_name)

            list_path = os.path.join(dataset_folder_base, "{}.txt".format(split_name))
            npy_path = os.path.join(dataset_folder_base, "{}.npy".format(split_name))

            list_paths.append([])
            with open(list_path, "r") as f_in:
                file_paths = f_in.readlines()
                for each_line in file_paths:
                    each_line = each_line.strip("\n")
                    list_paths[-1].append(each_line)

            img_instance_type_npy = np.load(npy_path)
            for img_instance_type in img_instance_type_npy:
                img_instance_types.append(img_instance_type)

        self.list_paths = list_paths
        self.img_instance_types = img_instance_types

        if "train" in split_name.lower():
            color_aug = nn.Sequential(
                MyRandomColorJitter(0.8, 1.2, 0.8, 1.2, 0.8, 1.2),
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.GaussianBlur(3, 1.5)]), p=0.1),
                transforms.Normalize(0.5, 0.5)
            )

            trans_aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRandomRotation(),
                transforms.Resize(image_size)
            )
        else:
            color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
            trans_aug = nn.Sequential(transforms.Resize(image_size))

        self.color_aug = color_aug
        self.trans_aug = trans_aug

    def __len__(self):
        return len(self.img_instance_types)

    def get_list_paths(self):
        """
        :return: the list of img paths
        """
        return self.list_paths

    def get_img_instance_types(self):
        """
        :return: the list of (dataset_idx, img_index, unique_index, type_this_index) pairs
        """
        return self.img_instance_types

    def __getitem__(self, idx):
        dataset_idx, img_idx, unique_index, type_this_idx = self.img_instance_types[idx]

        img = pil_loader(self.list_paths[dataset_idx][img_idx])
        img = self.transform(img)
        image = self.color_aug(img)

        folders = self.list_paths[dataset_idx][img_idx].split("/")

        mask_path = os.path.join(self.root_dir, "40x", self.dataset_names[dataset_idx],
                                 "mask", folders[11], folders[12][:-3] + "npy")

        mask = get_mask(mask_path, unique_index)
        mask = self.transform(mask)

        instance_now = crop_img_instance_with_bg(image, mask)

        image = self.trans_aug(instance_now)

        del img
        del mask

        type_this_idx = type_this_idx - dataset_mask[dataset_idx][0]

        return image, dataset_idx, type_this_idx


class LinearEvalSingleDatasetCrop(Dataset):
    """local dataset of 64*64"""

    def __init__(self, root_dir, dataset_names, dataset_idx, image_size=224, split_name="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_name = dataset_names[dataset_idx]

        dataset_name = dataset_names[dataset_idx]

        img_instance_types = []

        dataset_folder_base = os.path.join(root_dir, "40x", dataset_name)

        list_path = os.path.join(dataset_folder_base, "{}.txt".format(split_name))
        npy_path = os.path.join(dataset_folder_base, "{}.npy".format(split_name))

        list_paths = []
        with open(list_path, "r") as f_in:
            file_paths = f_in.readlines()
            for each_line in file_paths:
                each_line = each_line.strip("\n")
                list_paths.append(each_line)

        img_instance_type_npy = np.load(npy_path)
        for img_instance_type in img_instance_type_npy:
            img_instance_types.append(img_instance_type)

        self.list_paths = list_paths
        self.img_instance_types = img_instance_types
        self.dataset_idx = dataset_idx

        if "train" in split_name.lower():
            color_aug = nn.Sequential(
                MyRandomColorJitter(0.8, 1.2, 0.8, 1.2, 0.8, 1.2),
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.GaussianBlur(3, 1.5)]), p=0.1),
                transforms.Normalize(0.5, 0.5)
            )

            trans_aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRandomRotation(),
                transforms.Resize(image_size)
            )
        else:
            color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
            trans_aug = nn.Sequential(transforms.Resize(image_size))

        self.color_aug = color_aug
        self.trans_aug = trans_aug

    def __len__(self):
        return len(self.img_instance_types)

    def get_list_paths(self):
        """
        :return: the list of img paths
        """
        return self.list_paths

    def get_img_instance_types(self):
        """
        :return: the list of (dataset_idx, img_index, unique_index, type_this_index) pairs
        """
        return self.img_instance_types

    def __getitem__(self, idx):
        dataset_idx, img_idx, unique_index, type_this_idx = self.img_instance_types[idx]
        img = pil_loader(self.list_paths[img_idx])
        img = self.transform(img)
        image = self.color_aug(img)

        folders = self.list_paths[img_idx].split("/")

        mask_path = os.path.join(self.root_dir, "40x", self.dataset_name,
                                 "mask", folders[11], folders[12][:-3] + "npy")

        mask = get_mask(mask_path, unique_index)
        mask = self.transform(mask)

        instance_now = crop_img_instance_with_bg(image, mask)
        image = self.trans_aug(instance_now)

        del img
        del mask

        assert dataset_idx == self.dataset_idx, "dataset idx not correct!"

        return image, type_this_idx-dataset_mask[dataset_idx][0]


class LinearEvalMultiFCSupUnsupDatasetCrop(Dataset):
    """local dataset of 64*64"""

    def __init__(self, root_dir, dataset_names_sup, dataset_names_unsup, image_size=224,
                 split_name="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        if "train" in split_name.lower():
            self.dataset_names = dataset_names_sup+dataset_names_unsup
        else:
            self.dataset_names = dataset_names_sup

        list_paths = []
        img_instance_types = []
        for dataset_name in self.dataset_names:
            dataset_folder_base = os.path.join(root_dir, "40x", dataset_name)

            list_path = os.path.join(dataset_folder_base, "{}.txt".format(split_name))
            npy_path = os.path.join(dataset_folder_base, "{}.npy".format(split_name))

            list_paths.append([])
            with open(list_path, "r") as f_in:
                file_paths = f_in.readlines()
                for each_line in file_paths:
                    each_line = each_line.strip("\n")
                    list_paths[-1].append(each_line)

            img_instance_type_npy = np.load(npy_path)
            for img_instance_type in img_instance_type_npy:

                img_instance_types.append(img_instance_type)

        self.list_paths = list_paths
        self.img_instance_types = img_instance_types

        if "train" in split_name.lower():
            color_aug = nn.Sequential(
                MyRandomColorJitter(0.8, 1.2, 0.8, 1.2, 0.8, 1.2),
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.GaussianBlur(3, 1.5)]), p=0.1),
                transforms.Normalize(0.5, 0.5)
            )

            trans_aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRandomRotation(),
                transforms.Resize(image_size)
            )
        else:
            color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
            trans_aug = nn.Sequential(transforms.Resize(image_size))

        self.color_aug = color_aug
        self.trans_aug = trans_aug

    def __len__(self):
        return len(self.img_instance_types)

    def get_list_paths(self):
        """
        :return: get the list of img paths
        """
        return self.list_paths

    def get_img_instance_types(self):
        """
        :return: the list of (dataset_idx, img_index, unique_index, type_this_index) pairs
        """
        return self.img_instance_types

    def __getitem__(self, idx):
        dataset_idx, img_idx, unique_index, type_this_idx = self.img_instance_types[idx]

        img = pil_loader(self.list_paths[dataset_idx][img_idx])
        img = self.transform(img)
        image = self.color_aug(img)

        folders = self.list_paths[dataset_idx][img_idx].split("/")

        mask_path = os.path.join(self.root_dir, "40x", self.dataset_names[dataset_idx], "mask",
                                 folders[11], folders[12][:-3] + "npy")

        mask = get_mask(mask_path, unique_index)
        mask = self.transform(mask)

        instance_now = crop_img_instance_with_bg(image, mask)

        image = self.trans_aug(instance_now)

        del img
        del mask

        if dataset_idx < len(dataset_mask):
            type_this_idx = type_this_idx-dataset_mask[dataset_idx][0]
        else:
            type_this_idx = -1

        return image, dataset_idx, type_this_idx


class RandomApply(nn.Module):
    """
    class for random apply transformation
    """
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        """
        :param x: data to do augmentation
        :return: augmented data
        """
        if random.random() > self.p:
            return x
        return self.fn(x)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    """
    :param path: img_path
    :return: loaded img
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MyRandomRotation(torch.nn.Module):
    """Randomly rotate the tensor by 0, 90, 180, or 270"""
    def forward(self, x):
        """
        :param x: img to do random rotation
        :return: rotated img
        """
        angle = torch.randint(0, 4, size=(1, )).item()
        angle = angle * 90
        return TF.rotate(x, angle)


class MyRandomColorJitter(torch.nn.Module):
    """Randomly do the color jitter with specified range"""
    def __init__(self, brightness_min=0.7, brightness_max=1.3, contrast_min=0.7, contrast_max=2.0,
                 saturation_min=0.7, saturation_max=2.0):
        super().__init__()
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max

    def forward(self, img):
        """
        :param img: img to do augmentation
        :return: augmented image
        """
        fn_idx = torch.randperm(3)
        for fn_id in fn_idx:
            if fn_id == 0:
                brightness_factor = \
                    torch.tensor(1.0).uniform_(self.brightness_min, self.brightness_max).item()
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_brightness(img, brightness_factor)

            if fn_id == 1:
                contrast_factor = \
                    torch.tensor(1.0).uniform_(self.contrast_min, self.contrast_max).item()
                img = TF.adjust_contrast(img, contrast_factor)

            if fn_id == 2:
                saturation_factor = \
                    torch.tensor(1.0).uniform_(self.saturation_min, self.saturation_max).item()
                img = TF.adjust_saturation(img, saturation_factor)

        return img
