from __future__ import print_function, division
import os
import numpy as np
from train_valid_test_split_crop import crop_img_instance_with_bg
import cv2

print("train valid test split")
dataset_names = ["panNuke", "NuCLS", "CoNSeP", "MoNuSAC"]
# dataset_names = ['cpm15', 'cpm17', 'kumar', 'MoNuSeg', 'tnbc']
dataset_start_idx = 0

split_names = [["Fold_1", "Fold_2", "Fold_3"], ["QC"], ["Train", "Test"], ["train"]]
# split_names = [["Train"], ["Train", "Test"], ["train", "test_same", "test_diff"],
#                ["MoNuSegTestData"], ["Train"]]
dataset_label_start_indexes = [0, 5, 17, 24]
label_names = [["Neoplastic", "Inflammatory", "Connective", "Dead","Epithelial"],
               ['tumor', 'fibroblast', 'lymphocyte', 'plasma_cell', 'macrophage',
                'mitotic_figure', 'vascular_endothelium', 'myoepithelium',
                'apoptotic_body', 'neutrophil', 'ductal_epithelium', 'eosinophil'],
               ["other", "inflammatory", "healthy_epithelial", "malignant_epithelial",
                "fibroblast", "muscle", "endothelial"],
               ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]]
# label_names = [["None"], ["None"], ["None"], ["None"],  ["None"]]

def split_train_valid_test():
    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset_base = os.path.join("../nuclei-level-multi-class/40x",
                                    dataset_name)

        dataset_img_folder_base = os.path.join(
                                    "../nuclei-level-multi-class/40x",
                                    dataset_name,
                                    "img")
        dataset_type_base = os.path.join(
                                    "../nuclei-level-multi-class/40x",
                                    dataset_name,
                                    "type")
        dataset_mask_base = os.path.join(
                                    "../nuclei-level-multi-class/40x",
                                    dataset_name,
                                    "mask")
        dst_crop_base = os.path.join(
                                    "../nuclei-level-multi-class/40x",
                                    dataset_name,
                                    "crop")

        print("dataset base: ", dataset_base)

        os.makedirs(f"vis/{dataset_name}")
        train_list_path = os.path.join(dataset_base, "train_v2.txt")
        train_npy_path = os.path.join(dataset_base, "train_v2.npy")
        valid_list_path = os.path.join(dataset_base, "valid_v2.txt")
        valid_npy_path = os.path.join(dataset_base, "valid_v2.npy")
        test_list_path = os.path.join(dataset_base, "test_v2.txt")
        test_npy_path = os.path.join(dataset_base, "test_v2.npy")

        path_filename_list = []
        path_npy = []

        training_img_names = []
        training_folder = None
        validation_img_names = []
        validation_folder = None
        test_img_names = []
        test_folder = None
        
        if len(split_names[dataset_idx]) > 2:
            training_folder = split_names[dataset_idx][0]
            dataset_folder_path = os.path.join(dataset_img_folder_base, training_folder)
            training_img_names = os.listdir(dataset_folder_path)

            validation_folder = split_names[dataset_idx][1]
            dataset_folder_path = os.path.join(dataset_img_folder_base, validation_folder)
            validation_img_names = os.listdir(dataset_folder_path)

            test_folder = split_names[dataset_idx][2]
            dataset_folder_path = os.path.join(dataset_img_folder_base, test_folder)
            test_img_names = os.listdir(dataset_folder_path)

        elif len(split_names[dataset_idx]) == 2:
            training_folder = split_names[dataset_idx][0]
            validation_folder = split_names[dataset_idx][0]

            dataset_folder_path = os.path.join(dataset_img_folder_base, training_folder)
            img_names = os.listdir(dataset_folder_path)
            img_names_length = len(img_names)
            training_img_names = img_names[:int(img_names_length/10*8)]
            validation_img_names = img_names[int(img_names_length/10*8):]

            test_folder = split_names[dataset_idx][1]
            dataset_folder_path = os.path.join(dataset_img_folder_base, test_folder)
            test_img_names = os.listdir(dataset_folder_path)
        elif len(split_names[dataset_idx]) == 1:
            training_folder = split_names[dataset_idx][0]
            validation_folder = split_names[dataset_idx][0]
            test_folder = split_names[dataset_idx][0]

            dataset_folder_path = os.path.join(dataset_img_folder_base, training_folder)
            img_names = sorted(os.listdir(dataset_folder_path))
            img_names_length = len(img_names)
            training_img_names = img_names[:int(img_names_length / 10 * 7)]
            validation_img_names = img_names[int(img_names_length / 10 * 7):int(img_names_length / 10 * 8)]
            test_img_names = img_names[int(img_names_length / 10 * 8):]
            dataset_folder_path = os.path.join(dataset_img_folder_base, test_folder)

        for (split, dst_split, split_files) in zip([training_folder, validation_folder, test_folder],
                                        ["training", "validation", "test"],
                                        [training_img_names, validation_img_names, test_img_names]):
            path_filename_list.append([])
            path_npy.append([])

            dataset_folder_path = os.path.join(dataset_img_folder_base, split)
            # dst_folder = os.path.join(dst_crop_base, dst_split)

            for img_idx, img_name in enumerate(split_files):
                img_filepath = os.path.join(dataset_img_folder_base, split, img_name)
                img = cv2.imread(img_filepath)
                path_filename_list[-1].append(img_filepath)
                # path_filename_list.append(img_filepath)
                
                type_filepath = os.path.join(dataset_type_base, split, img_name[:-3] + "npy")
                types_this_img = np.load(type_filepath)

                mask_filepath = os.path.join(dataset_mask_base, split, img_name[:-3] + "npy")
                mask_this_img = np.load(mask_filepath)

                instances = sorted(np.unique(mask_this_img))[1:]
                for (instance_idx, instance_type) in types_this_img:
                # for instance_idx in instances:
                    instance = crop_img_instance_with_bg(img, mask_this_img, target_label=instance_idx)
                    if instance is not None:
                        path_npy[-1].append([dataset_idx+dataset_start_idx, img_idx, instance_idx, instance_type+dataset_label_start_indexes[dataset_idx]])
                        # path_npy.append([dataset_idx+dataset_start_idx, len(path_filename_list)-1, instance_idx, -1])
                    
                        cv2.imwrite(f"vis/{dataset_name}/{img_name[:-4]}_{instance_idx}.png", instance)

            # path_npy = np.asarray(path_npy)
        
        path_npy = np.asarray(path_npy)

        with open(train_list_path, "w") as f:
            for train_file_path in path_filename_list[0]:
                f.write(train_file_path+"\n")

        with open(train_npy_path, "wb") as f:
            np.save(f, path_npy[0])

        print("traing img length: ", len(path_filename_list[0]))
        print("training shape: ", path_npy[0].shape)
        print("unique labels: ", np.unique(path_npy[0][:, -1]))

        with open(valid_list_path, "w") as f:
            for valid_file_path in path_filename_list[1]:
                f.write(valid_file_path + "\n")

        with open(valid_npy_path, "wb") as f:
            np.save(f, path_npy[1])
        
        print("traing img length: ", len(path_filename_list[1]))
        print("validation shape: ", path_npy[1].shape)
        print("unique labels: ", np.unique(path_npy[1][:, -1]))

        with open(test_list_path, "w") as f:
            for test_file_path in path_filename_list[2]:
                f.write(test_file_path + "\n")

        with open(test_npy_path, "wb") as f:
            np.save(f, path_npy[2])
        
        print("traing img length: ", len(path_filename_list[2]))
        print("test shape: ", path_npy[2].shape)
        print("unique labels: ", np.unique(path_npy[2][:, -1]))
        """

        with open(train_list_path, "w") as f:
            for train_file_path in path_filename_list:
                f.write(train_file_path + "\n")

        with open(train_npy_path, "wb") as f:
            np.save(f, path_npy)

        print("train list path: ", len(path_filename_list))
        print("training shape: ", path_npy.shape)

        with open(valid_list_path, "w") as f:
            pass

        with open(valid_npy_path, "wb") as f:
            pass

        with open(test_list_path, "w") as f:
            pass

        with open(test_npy_path, "wb") as f:
            pass
        """

def file_counting():
    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset_base = os.path.join("../nuclei-level-multi-class/40x",
                                    dataset_name)

        dst_crop_base = os.path.join(
                                    "../nuclei-level-multi-class/40x",
                                    dataset_name,
                                    "crop")

        for split_name in os.listdir(dst_crop_base):

            split_path = os.path.join(dst_crop_base, split_name)

            for folder_name in os.listdir(split_path):
                folder_path = os.path.join(split_path, folder_name)

                file_names = os.listdir(folder_path)
            
                print(dataset_name, "\t", split_name, "\t", folder_name, "\t", len(file_names))

        input("checkpoint")


if __name__ == "__main__":
    split_train_valid_test()
    # file_counting()