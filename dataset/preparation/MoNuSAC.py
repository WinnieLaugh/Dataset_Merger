import numpy as np
import os
import cv2
import scipy.io
from utils.utils import draw_dilation, get_full_mask
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import scipy.io
import openslide

splits = ['Train', 'Test']

nucleus_label_codebook_to_code = {4: "Epithelial", 25: "Lymphocyte", 9: "Neutrophil", 26: "Macrophage"}

nucleus_label_codebook = {"Epithelial": 0, "Lymphocyte": 1, "Neutrophil": 2, "Macrophage": 3}


def process_ori():
    for split in splits:
        img_folder = f"../nuclei-level/40x/MoNuSAC/{split}"

        dst_img_folder = "../nuclei-level-cleaned/40x/MoNuSAC/img/{}".format(split)
        dst_instance_mask_folder = "../nuclei-level-cleaned/40x/MoNuSAC/mask/{}".format(split)
        dst_type_folder = "../nuclei-level-cleaned/40x/MoNuSAC/type/{}".format(split)
        dst_overlay_folder = "../nuclei-level-cleaned/40x/MoNuSAC/overlay/{}".format(split)

        os.makedirs(dst_img_folder, exist_ok=True)
        os.makedirs(dst_instance_mask_folder, exist_ok=True)
        os.makedirs(dst_type_folder, exist_ok=True)
        os.makedirs(dst_overlay_folder, exist_ok=True)

        for img_folder_inside in tqdm(os.listdir(img_folder)):
            img_folder_inside = os.path.join(img_folder, img_folder_inside)

            for img_name in sorted(os.listdir(img_folder_inside)):
                if img_name[-3:] == "png":
                    img_filepath = os.path.join(img_folder_inside, img_name)
                    img = cv2.imread(img_filepath)

                    mask_filepath = os.path.join(img_folder_inside, img_name[:-3] + "xml")
                    tree = ET.parse(mask_filepath)
                    root = tree.getroot()

                    instance_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)
                    instance_type = []

                    unique_label_now = 1

                    for idx, instance in enumerate(root.iter("Annotation")):
                        type_this_one = -1
                        for child in instance:
                            if child.tag == 'Attributes':
                                type_this_one = child[0].attrib['Name']
                                if type_this_one not in nucleus_label_codebook.keys():
                                    type_this_one = -2
                                else:
                                    type_this_one = nucleus_label_codebook[type_this_one]

                            if type_this_one == -2:
                                continue

                            if child.tag == 'Regions':
                                for child_child in child:
                                    if child_child.tag == "Region":
                                        around_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)
                                        for child_child_child in child_child:
                                            if child_child_child.tag == "Vertices":
                                                for vertex in child_child_child:
                                                    x = int(float(vertex.attrib['X']))
                                                    y = int(float(vertex.attrib['Y']))
                                                    if y < instance_mask.shape[0] and x < instance_mask.shape[1]:
                                                        around_mask[y, x] = unique_label_now

                                        around_mask = get_full_mask(around_mask, unique_label_now)
                                        if around_mask is not None:
                                            around_mask = get_full_mask(around_mask, unique_label_now)

                                        if around_mask is not None:
                                            indexes_this_unique_label = np.where(around_mask == unique_label_now)
                                            instance_mask[indexes_this_unique_label] = unique_label_now

                                        instance_type.append([unique_label_now, type_this_one])
                                        unique_label_now += 1

                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    dst_img_path = os.path.join(dst_img_folder, img_name[:-3] + "png")
                    cv2.imwrite(dst_img_path, img)

                    dst_instance_mask_path = os.path.join(dst_instance_mask_folder, img_name[:-3] + "npy")
                    with open(dst_instance_mask_path, "wb") as f:
                        np.save(f, instance_mask)

                    dst_type_path = os.path.join(dst_type_folder, img_name[:-3] + "npy")
                    instance_type = np.asarray(instance_type)
                    with open(dst_type_path, "wb") as f:
                        np.save(f, instance_type)

                    overlay_img = draw_dilation(img, instance_mask, instance_type)
                    dst_overlay_path = os.path.join(dst_overlay_folder, img_name[:-3] + "png")
                    cv2.imwrite(dst_overlay_path, overlay_img)


if __name__ == '__main__':
    process_ori()
