"""
Inference class. Put your data in the folder 'example_data', including image and mask.
"""

import os
import sys
sys.path.append('./')

from argparse import ArgumentParser
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from dataset.dataset import InferenceDatasetCrop
from model.utils import get_preds_inference, draw_dilation, dataset_mask
from model.model import MultiLabelModelLabelCompletion

parser = ArgumentParser()
parser.add_argument("--name", type=str, default="test1", help="name of experiment")
parser.add_argument("--root_dir", type=str, default="example_data/CoNSeP",
                    help="the data folder where the inference data are stored")
parser.add_argument("--img_name", type=str, default="test_11.png", help="img name to inference")

args = parser.parse_args()


def visualize():
    """
    function to visualize predicted results.
    overlay will be stored in 'result' folder.
    """
    device = torch.device(0)

    dataset_names = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]
    imagenet_test = InferenceDatasetCrop(args.root_dir, args.img_name)

    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test, batch_size=1,
                                                           num_workers=0)

    resnet = resnet50(num_classes=35)
    classifiers = [nn.Sequential(nn.Linear(35 - len(dataset_mask[_]), 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, len(dataset_mask[_])))  \
                                    for _ in range(len(dataset_mask))]
    model = MultiLabelModelLabelCompletion(resnet=resnet, classifiers=classifiers)

    checkpoint_path = "checkpoints/{}/label_completion/improved-net_latest.pt".format(args.name)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    instance_preds = [[] for _ in range(len(dataset_mask))]

    with torch.no_grad():
        model.eval()

        for (image, unique_indexes) in imagenet_test_dataloader:
            image = image.to(device).float()
            pred, out = model(image)

            _, _, real_pred_instances = get_preds_inference(pred, out)

            for dataset_idx in range(len(dataset_mask)):
                instance_preds[dataset_idx].append([unique_indexes[0], \
                                                    real_pred_instances[dataset_idx][0]])

    img_path = os.path.join(args.root_dir, args.img_name)
    img = cv2.imread(img_path)
    mask_path = os.path.join(args.root_dir, args.img_name[:-4]+"_mask.npy")
    mask = np.load(mask_path)

    os.makedirs("result", exist_ok=True)
    for dataset_idx in range(len(dataset_mask)):
        instance_pred = np.array(instance_preds[dataset_idx], dtype=np.int)
        overlay = draw_dilation(img, mask, instance_pred)
        overlay_path = os.path.join("result", \
                                    args.img_name[:-4] + "pred_" + \
                                    dataset_names[dataset_idx] + "_overlay.png")
        cv2.imwrite(overlay_path, overlay)

if __name__ == "__main__":
    print("#################################################")
    visualize()
