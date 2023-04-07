import os
import cv2
import numpy as np
from utils.utils import draw_dilation, img_mask_alignment, check_if_rectangle
from tqdm import tqdm
import scipy.io

nucleus_label_codebook = {'fov':   -1,                 'tumor': 0,          'fibroblast':   1,
                   'lymphocyte':	2,           'plasma_cell': 3,          'macrophage':	4,
               'mitotic_figure':	5,  'vascular_endothelium': 6,       'myoepithelium':	7,
               'apoptotic_body':	8,            'neutrophil':	9,  'ductal_epithelium':	10,
                   'eosinophil':   11,             'unlabeled':	-1}

nucleus_label_codebook_this = {253: 'fov',             1: 'tumor',                      2: 'fibroblast',
                                 3: 'lymphocyte',      4: 'plasma_cell',                5: 'macrophage',
                                 6: 'mitotic_figure',  7: 'vascular_endothelium',       8: 'myoepithelium',
                                 9: 'apoptotic_body', 10: 'neutrophil',                 11: 'ductal_epithelium',
                                12: 'eosinophil',     99: 'unlabeled'}

### in reality fov equals 255, unlabeled equals 0

img_folder = "../nuclei-level/40x/NuCLS/QC/rgb"
mask_folder = "../nuclei-level/40x/NuCLS/QC/mask"
src_instance_mask_folder = "../nuclei-level/40x/NuCLS/QC/hovernet/QC"

dst_img_folder = "../nuclei-level-cleaned/40x/NuCLS/img/QC"
dst_instance_mask_folder = "../nuclei-level-cleaned/40x/NuCLS/mask/QC"
dst_type_folder = "../nuclei-level-cleaned/40x/NuCLS/type/QC"
dst_overlay_folder = "../nuclei-level-cleaned/40x/NuCLS/overlay/QC"

os.makedirs(dst_img_folder, exist_ok=True)
os.makedirs(dst_instance_mask_folder, exist_ok=True)
os.makedirs(dst_type_folder, exist_ok=True)
os.makedirs(dst_overlay_folder, exist_ok=True)

instance_number = 0
picked_instance_number = 0
for img_filename in tqdm(os.listdir(img_folder)):
    img_path = os.path.join(img_folder, img_filename)
    img = cv2.imread(img_path)

    mask_path = os.path.join(mask_folder, img_filename)
    mask = cv2.imread(mask_path)

    instance_mask_path = os.path.join(src_instance_mask_folder, img_filename[:-3]+"mat")
    mat = scipy.io.loadmat(instance_mask_path)
    instance_mask = mat['inst_map']

    img, mask = img_mask_alignment(img, mask)
    type_mask = mask[:, :, 2]

    background_indexes = np.where(type_mask == 253)
    type_mask[background_indexes] = 0

    unlabeled_indexes = np.where(type_mask == 99)
    type_mask[unlabeled_indexes] = 0

    instance_type = []

    for instance_idx in np.sort(np.unique(instance_mask)):
        instance_indexes = np.where(instance_mask == instance_idx)

        label_this_instance, counts = np.unique(type_mask[instance_indexes], return_counts=True)
        if len(label_this_instance) > 1:
            instance_number += 1
            ind = np.argmax(counts)
            label_this_instance = label_this_instance[ind]
            if counts[ind] < len(instance_indexes[0]) * 0.5:
                instance_mask[instance_indexes] = 0
                continue
        else:
            label_this_instance = label_this_instance[0]

        if label_this_instance == 0:
            instance_mask[instance_indexes] = 0
            continue
        else:
            instance_type_this = nucleus_label_codebook_this[label_this_instance]
            label_this_instance = nucleus_label_codebook[instance_type_this]

        picked_instance_number += 1
        instance_type.append([instance_idx, label_this_instance])

        print("picked instance number: ", picked_instance_number)
        print("instance number: ", instance_number)

    instance_type = np.asarray(instance_type)
    img_dilated = draw_dilation(img, instance_mask, instance_type)

    dst_img_path = os.path.join(dst_img_folder, img_filename[:-3]+"png")
    cv2.imwrite(dst_img_path, img)

    dst_instance_mask_path = os.path.join(dst_instance_mask_folder, img_filename[:-3] + "npy")
    with open(dst_instance_mask_path, "wb") as f:
        np.save(f, instance_mask)

    dst_type_path = os.path.join(dst_type_folder, img_filename[:-3]+"npy")
    with open(dst_type_path, "wb") as f:
        np.save(f, instance_type)

    dst_overlay_path = os.path.join(dst_overlay_folder, img_filename)
    cv2.imwrite(dst_overlay_path, img_dilated)

