import numpy as np
import os
import cv2
import scipy.io
from utils.utils import draw_dilation
from tqdm import tqdm
import pickle
import cv2

splits = ["Train", "Test"]

def process_ori():
    for split in splits:
        img_folder = "../nuclei-level/40x/consep/CoNSeP/{}/Images".format(split)
        mask_folder = "../nuclei-level/40x/consep/CoNSeP/{}/Labels".format(split)

        dst_img_folder = "../nuclei-level-cleaned/40x/CoNSeP/img/{}".format(split)
        dst_instance_mask_folder = "../nuclei-level-cleaned/40x/CoNSeP/mask/{}".format(split)
        dst_type_folder = "../nuclei-level-cleaned/40x/CoNSeP/type/{}".format(split)
        dst_overlay_folder = "../nuclei-level-cleaned/40x/CoNSeP/overlay/{}".format(split)

        os.makedirs(dst_img_folder, exist_ok=True)
        os.makedirs(dst_instance_mask_folder, exist_ok=True)
        os.makedirs(dst_type_folder, exist_ok=True)
        os.makedirs(dst_overlay_folder, exist_ok=True)

        nucleus_label_codebook = {"other": 0, "inflammatory": 1, "healthy_epithelial": 2, "malignant_epithelial": 3,
                                  "fibroblast": 4, "muscle": 5, "endothelial": 6}

        nucleus_label_codebook_this = {"other": 1, "inflammatory": 2, "healthy_epithelial": 3, "malignant_epithelial": 4,
                                       "fibroblast": 5, "muscle": 6, "endothelial": 7}

        for src_mat_file in tqdm(os.listdir(img_folder)):
            img_file = src_mat_file[:-3] + "png"
            img_filepath = os.path.join(img_folder, img_file)
            img = cv2.imread(img_filepath)

            mask_filepath = os.path.join(mask_folder, img_file[:-3] + "mat")
            mat = scipy.io.loadmat(mask_filepath)

            instance_mask = mat['inst_map']
            instance_type = mat['inst_type']

            instance_type_cleaned = np.zeros((len(instance_type), 2), dtype=int)

            for label_this in nucleus_label_codebook_this.keys():
                instance_indexes = np.where(instance_type == nucleus_label_codebook_this[label_this])
                instance_type_cleaned[instance_indexes, 1] = nucleus_label_codebook[label_this]

            for idx in range(len(instance_type_cleaned)):
                instance_type_cleaned[idx, 0] = idx + 1

            overlay_img = draw_dilation(img, instance_mask, instance_type=instance_type_cleaned)

            dst_img_path = os.path.join(dst_img_folder, img_file)
            cv2.imwrite(dst_img_path, img)

            dst_instance_mask_path = os.path.join(dst_instance_mask_folder, img_file[:-3] + "npy")
            with open(dst_instance_mask_path, "wb") as f:
                np.save(f, instance_mask)

            dst_type_path = os.path.join(dst_type_folder, img_file[:-3] + "npy")
            with open(dst_type_path, "wb") as f:
                np.save(f, instance_type_cleaned)

            dst_overlay_path = os.path.join(dst_overlay_folder, img_file)
            cv2.imwrite(dst_overlay_path, overlay_img)


if __name__ == '__main__':
    process_ori()

