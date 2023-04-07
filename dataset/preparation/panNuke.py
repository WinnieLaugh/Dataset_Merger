import numpy as np
import os
import cv2
import scipy.io
from utils.utils import draw_dilation
from tqdm import tqdm

splits = ['Fold_1', 'Fold_2', 'Fold_3']

nucleus_label_codebook = {"Neoplastic": 0, "Inflammatory": 1, "Connective": 2, "Dead": 3, "Epithelial": 4}

nucleus_label_codebook_this = {0: "Neoplastic", 1: "Inflammatory", 2: "Connective", 3: "Dead", 4: "Epithelial"}


def process_ori():
    for split in splits:
        img_folder = "../nuclei-level/40x/panNuke/{}/images".format(split)
        mask_folder = "../nuclei-level/40x/panNuke/{}/masks".format(split)

        dst_img_folder = "../nuclei-level-cleaned/40x/panNuke/img/{}".format(split)
        dst_instance_mask_folder = "../nuclei-level-cleaned/40x/panNuke/mask/{}".format(split)
        dst_type_folder = "../nuclei-level-cleaned/40x/panNuke/type/{}".format(split)
        dst_overlay_folder = "../nuclei-level-cleaned/40x/panNuke/overlay/{}".format(split)

        os.makedirs(dst_img_folder, exist_ok=True)
        os.makedirs(dst_instance_mask_folder, exist_ok=True)
        os.makedirs(dst_type_folder, exist_ok=True)
        os.makedirs(dst_overlay_folder, exist_ok=True)

        img_filepath = os.path.join(img_folder, "images.npy")
        imgs = np.load(img_filepath)

        mask_filepath = os.path.join(mask_folder, "masks.npy")
        masks = np.load(mask_filepath)

        for img_idx in tqdm(range(imgs.shape[0])):
            img = imgs[img_idx]
            mask = masks[img_idx]
            img = np.asarray(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            instance_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)
            instance_type = []
            for channel_idx in range(5):
                unique_labels = np.unique(mask[:, :, channel_idx])
                for unique_label in sorted(unique_labels)[1:]:
                    indexes = np.where(mask[:, :, channel_idx] == unique_label)
                    instance_mask[indexes] = int(unique_label)
                    instance_type.append(
                        (int(unique_label), nucleus_label_codebook[nucleus_label_codebook_this[channel_idx]]))

            instance_type = np.array(instance_type)
            overlay_img = draw_dilation(img, instance_mask, instance_type)

            dst_img_path = os.path.join(dst_img_folder, "{:04d}.png".format(img_idx))
            cv2.imwrite(dst_img_path, img)

            dst_instance_mask_path = os.path.join(dst_instance_mask_folder, "{:04d}.npy".format(img_idx))
            with open(dst_instance_mask_path, "wb") as f:
                np.save(f, instance_mask)

            dst_type_path = os.path.join(dst_type_folder, "{:04d}.npy".format(img_idx))
            with open(dst_type_path, "wb") as f:
                np.save(f, instance_type)

            dst_overlay_path = os.path.join(dst_overlay_folder, "{:04d}.png".format(img_idx))
            cv2.imwrite(dst_overlay_path, overlay_img)

if __name__ == '__main__':
    process_ori()
