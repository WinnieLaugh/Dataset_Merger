"""
Utils for dataset loading, loss function, accuracy calculation, and etc.
"""
import os
import random
import PIL
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

dataset_mask = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # NuCLS
                [12, 13, 14, 15, 16, 17, 18], # BreCaHAD
                [19, 20, 21, 22, 23, 24, 25], # CoNSeP
                [26, 27, 28, 29], # MoNuSAC
                [30, 31, 32, 33, 34]] # panNuke

label_colors = [[  0,   0,   0], [255, 255,   0],
                [255,   0,   0], [255,   0, 255],
                [  0, 255,   0], [  0, 255, 255],
                [  0,   0, 255]]

softmax = nn.Softmax(dim=1)
cross_entropy = nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss(reduction='mean')
mse_loss_sum = torch.nn.MSELoss(reduction='sum')


def draw_dilation(img, instance_mask, instance_type=None):
    """
    :param img: original image
    :param instance_mask: mask for the instances
    :param instance_type: types of each instance
    :return: overlay image
    """
    img_overlay = img.copy()

    if instance_type is not None:
        for instance in instance_type:
            binary_map = np.zeros_like(img, dtype=np.uint8)

            indexes = np.where(instance_mask == instance[0])

            binary_map[indexes] = 255
            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

            img_overlay[inst_pixels_dilated] = label_colors[(instance[1]%len(label_colors))]
            img_overlay[indexes] = img[indexes]
    else:
        for instance_idx in sorted(np.unique(instance_mask))[1:]:
            binary_map = np.zeros_like(img, dtype=np.uint8)
            indexes = np.where(instance_mask == instance_idx)

            binary_map[indexes] = 255
            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

            img_overlay[inst_pixels_dilated] = [random.randint(0, 255), \
                                                random.randint(0, 255), \
                                                random.randint(0, 255)]
            img_overlay[indexes] = img[indexes]

    return img_overlay


def loss_fn(pred, dataset_idxes, type_idxes):
    """
    :param pred: output from the sparse training
    :param dataset_idxes: dataset indexes of the input instances
    :param type_idxes: ground truth types of the input instances
    :return: loss of this batch
    """
    loss = 0

    pred = torch.clamp(pred, min=-10, max=10)
    pred = torch.exp(pred)

    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        if indexes.shape[0] > 0:
            indexes = indexes.squeeze(1)
            frac_up = pred[indexes, type_idxes[indexes]+dataset_mask[dataset_idx][0]]
            frac_down = torch.sum(pred[indexes, \
                                  dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1]+1], \
                                  dim=1)
            loss += torch.sum(- torch.log(frac_up / frac_down))

    return loss / pred.shape[0]


def get_accuracy(pred, dataset_idxes, type_idxes):
    """
    :param pred: pred result of the sparse training
    :param dataset_idxes: dataset indexes of the input instances
    :param type_idxes: ground truth types of the input instances
    :return: accuracy of each dataset, and the number of instances of each dataset
    """
    pred = torch.exp(pred)

    total_num = torch.zeros(len(dataset_mask)).to(dataset_idxes.device)
    correct_array = torch.zeros(len(dataset_mask)).to(dataset_idxes.device)

    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        if indexes.shape[0] > 0:
            indexes = indexes.squeeze(1)
            pred_this = pred[indexes, dataset_mask[dataset_idx][0]: dataset_mask[dataset_idx][-1]+1]

            __, pred_this = torch.max(pred_this, 1)
            correct_bool = (pred_this == type_idxes[indexes])

            correct_array[dataset_idx] = correct_bool.sum().item()
            total_num[dataset_idx] = len(indexes)

    return correct_array, total_num


def get_pred_true_f1(pred, dataset_idxes, type_idxes):
    """
    :param pred: output from the sparse training
    :param dataset_idxes: dataset indexes of the input instances
    :param type_idxes: types of the input instances
    :return: correct_array: the correct predictions of each dataset in this batch
             total_num: the number of instances of each dataset in this batch
             y_pred: the prediction of each instance
             y_gt: the ground truth of each instance
    """
    pred = torch.exp(pred)
    total_num = torch.zeros(len(dataset_mask))
    correct_array = torch.zeros(len(dataset_mask))

    y_pred = []
    y_gt = []

    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        indexes = indexes.squeeze(1)

        if indexes.shape[0] > 0:
            pred_this = pred[indexes, \
                        dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1] + 1]
            gt_this = type_idxes[indexes]
            _, pred_this = torch.max(pred_this, 1)

            correct_array[dataset_idx] = (pred_this == gt_this).sum().item()
            total_num[dataset_idx] = len(indexes)

            y_pred.append(pred_this.tolist())
            y_gt.append(gt_this.tolist())
        else:
            y_pred.append([])
            y_gt.append([])

    return correct_array, total_num, y_pred, y_gt


def get_other_probabilities(preds, dataset_idx):
    """
    :param preds: the prediction of all the labels
    :param dataset_idx: the dataset indexes of the instances
    :return: the predictions of all other labels except its own label
    """
    preds = torch.transpose(preds, 0, 1)

    col_indexes = list(range(preds.shape[0]))
    for label_idx in dataset_mask[dataset_idx]:
        col_indexes.remove(label_idx)

    preds_out = preds[col_indexes]
    preds_out = torch.transpose(preds_out, 0, 1)

    return preds_out


def loss_fn_correlation_learning(output, type_idxes):
    """
    :param output: the prediction of the label correlation
    :param type_idxes: ground truth types of the instances
    :return: loss of this batchs
    """
    loss_1 = cross_entropy(output, type_idxes)

    return loss_1


def get_accuracy_correlation_learning(pred, output, type_idxes, f1_calculation=False):
    """
    :param pred: predictions from the base resnet50
    :param output: predictions from the label correlation layers
    :param type_idxes: ground truth types of the instances
    :param f1_calculation: whether to compute f1 or not
    :return: if not to compute weighted f1:
            return correct_array: the number of correct predictions for each dataset in this batch.
            total_num: the number of instances of each dataset in this batch
            if to computer weighted f1, also return:
             pred_this: predictions for each instance of base resnet50
             output_this: predictions for each instance of label correlation layers
             real_this: predictions of the mean probabilities for each instance
             type_idxes: ground truth types for each instance
    """
    pred = torch.exp(pred)
    output = torch.exp(output)

    correct_array = torch.zeros(3)

    __, pred_this = torch.max(pred, 1)
    correct_bool = (pred_this == type_idxes)
    correct_array[0] = correct_bool.sum().item()

    __, output_this = torch.max(output, 1)
    correct_bool = (output_this == type_idxes)
    correct_array[1] = correct_bool.sum().item()

    total_num = type_idxes.shape[0]

    real = pred + output
    _, real_this = torch.max(real, 1)
    correct_bool = (real_this == type_idxes)
    correct_array[2] = correct_bool.sum().item()

    pred_this = pred_this.tolist()
    output_this = output_this.tolist()
    real_this = real_this.tolist()
    type_idxes = type_idxes.tolist()

    if f1_calculation:
        return correct_array, total_num, pred_this, output_this, real_this, type_idxes

    return correct_array, total_num


def get_multiple_probabilities(preds):
    """
    :param preds: predictions for all the labels
    :return: 5 different groups of predictions, each with one set of labels absent
    """
    pred_out = []

    for dataset_idx, _ in enumerate(dataset_mask):
        pred_input = torch.transpose(preds, 0, 1)

        col_indexes = list(range(pred_input.shape[0]))
        for label_idx in dataset_mask[dataset_idx]:
            col_indexes.remove(label_idx)

        x_out_single = pred_input[col_indexes]
        x_out_single = torch.transpose(x_out_single, 0, 1)
        pred_out.append(x_out_single)

    return pred_out


def get_accuracy_label_completion(pred, output, dataset_idxes, type_idxes, f1_calculation=False):
    """
    :param pred: predictions from the base resnet50
    :param output: predictions from the label correlation layers
    :param dataset_idxes: dataset indexes of the instances in the batch
    :param type_idxes: ground truth types of the instances in the batch
    :param f1_calculation: whether to compute f1 or not
    :return: if not to compute weighted f1:
            return correct_array: the number of correct predictions for each dataset in this batch.
            total_num: the number of instances of each dataset in this batch
            if to computer weighted f1, also return:
             pred_instances: predictions for each instance of base resnet50
             output_instances: predictions for each instance of label correlation layers
             real_this: predictions of the mean probabilities for each instance
             type_idxes: ground truth types for each instance
    """
    correct_array = torch.zeros((len(dataset_mask), 3))
    total_num = torch.zeros((len(dataset_mask),))
    pred_instances = [[] for _ in range(len(dataset_mask))]
    output_instances = [[] for _ in range(len(dataset_mask))]
    ground_truth_this = [[] for _ in range(len(dataset_mask))]
    real_pred_instances = [[] for _ in range(len(dataset_mask))]

    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        indexes = indexes.squeeze(1)

        if indexes.shape[0] > 0:
            pred_single = pred[indexes, \
                          dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1] + 1]
            output_single = output[dataset_idx][indexes]
            pred_single = softmax(pred_single)
            output_single = softmax(output_single)

            real_pred = pred_single + output_single

            __, pred_this = torch.max(pred_single, 1)
            correct_bool = (pred_this == type_idxes[indexes])
            correct_array[dataset_idx, 0] = correct_bool.sum().item()

            __, output_this = torch.max(output_single, 1)
            correct_bool = (output_this == type_idxes[indexes])
            correct_array[dataset_idx, 1] = correct_bool.sum().item()

            _, real_pred_this = torch.max(real_pred, 1)
            correct_bool = (real_pred_this == type_idxes[indexes])
            correct_array[dataset_idx, 2] = correct_bool.sum().item()

            if f1_calculation:
                pred_instances[dataset_idx] = pred_this.tolist()
                output_instances[dataset_idx] = output_this.tolist()
                real_pred_instances[dataset_idx] = real_pred_this.tolist()
                ground_truth_this[dataset_idx] = (type_idxes[indexes]).tolist()

            total_num[dataset_idx] = indexes.shape[0]

    if f1_calculation:
        return correct_array, total_num, pred_instances, \
               output_instances, real_pred_instances, ground_truth_this

    return correct_array, total_num


def loss_fn_label_completion(pred, output, dataset_idxes, type_idxes, threshold=0.99):
    """
    Cross entropy loss of the ground truth labels
    :param pred: predictions from the base resnet50
    :param output: predictions from the label correlation layers
    :param dataset_idxes: dataset indexes of the instances in the batch
    :param type_idxes: ground truth types of the instances in the batch
    :param threshold: confidence threshold of label completion,
    :return: loss of this batch, loss of cross entropy with ground truth labels in this batch,
            loss of cross entropy with completed pseudo labels in this batch,
            number of instances of each dataset in this batch
    """
    loss_0 = torch.zeros((len(dataset_mask), 2)).to(pred.device)
    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        indexes = indexes.squeeze(1)

        if indexes.shape[0] > 0:
            pred_this_loss = cross_entropy(
                pred[indexes, dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1] + 1],
                type_idxes[indexes])
            output_this_loss = cross_entropy(output[dataset_idx][indexes], type_idxes[indexes])
            loss_0[dataset_idx, 0] += pred_this_loss
            loss_0[dataset_idx, 1] += output_this_loss

    # one hot cross entropy
    loss_1 = torch.zeros((len(dataset_mask), 2), dtype=torch.float).to(pred.device)
    used_length = torch.zeros((len(dataset_mask), 2), dtype=torch.int).to(pred.device)
    all_indexes = range(type_idxes.shape[0])

    # indexes = all_indexes
    for dataset_idx, _ in enumerate(dataset_mask):
        indexes = torch.nonzero((dataset_idxes == dataset_idx), as_tuple=False)
        indexes = indexes.squeeze(1).detach().cpu().numpy()

        indexes = list(set(all_indexes) - set(indexes))

        pred_single = pred[indexes, dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1] + 1]
        pred_single_softmax = softmax(pred_single)
        output_single = output[dataset_idx][indexes]
        output_single_softmax = softmax(output_single)

        __, pred_this = torch.max(pred_single, 1)
        __, output_this = torch.max(output_single, 1)

        indexes_select = range(pred_single.shape[0])
        pred_single_certain = (pred_single_softmax[indexes_select, \
                                                   pred_this[indexes_select]] > threshold)
        output_single_certain = (output_single_softmax[indexes_select, \
                                                       output_this[indexes_select]] > threshold)

        output_certain_select_indexes = torch.nonzero(output_single_certain, as_tuple=False)
        output_certain_select_indexes = output_certain_select_indexes.squeeze(1)
        if output_certain_select_indexes.shape[0] > 0:
            pred_this_loss = cross_entropy(pred_single[output_certain_select_indexes],
                                             output_this[output_certain_select_indexes].detach())
            loss_1[dataset_idx, 0] += pred_this_loss
            used_length[dataset_idx, 0] += output_certain_select_indexes.shape[0]

        pred_certain_select_indexes = torch.nonzero(pred_single_certain, as_tuple=False)
        pred_certain_select_indexes = pred_certain_select_indexes.squeeze(1)
        if pred_certain_select_indexes.shape[0] > 0:
            output_this_loss = cross_entropy(output_single[pred_certain_select_indexes],
                                           pred_this[pred_certain_select_indexes].detach())
            loss_1[dataset_idx, 1] += output_this_loss
            used_length[dataset_idx, 1] += pred_certain_select_indexes.shape[0]

    return loss_0.mean() + loss_1.mean(), loss_0.mean(), loss_1.mean(), used_length


def get_preds_inference(pred, output):
    """
    :param pred: prediction scores from the base resnet50
    :param output: prediction scores from the label correlation layers
    :return: prediction from base resnet50 for each instance,
             prediction from label correlation layers for each instance,
             prediction of mean probabilities for each instance
    """
    pred_instances = []
    output_instances = []
    real_pred_instances = []

    indexes = range(pred.shape[0])

    for dataset_idx, _ in enumerate(dataset_mask):
        pred_single = pred[indexes, dataset_mask[dataset_idx][0]:dataset_mask[dataset_idx][-1] + 1]
        output_single = output[dataset_idx][indexes]
        pred_single = softmax(pred_single)
        output_single = softmax(output_single)

        __, pred_this = torch.max(pred_single, 1)
        __, output_this = torch.max(output_single, 1)

        real_pred = (pred_single + output_single) / 2
        _, real_pred_this = torch.max(real_pred, 1)

        pred_this = pred_this.tolist()
        output_this = output_this.tolist()
        real_pred_this = real_pred_this.tolist()
        pred_instances.append(pred_this)
        output_instances.append(output_this)
        real_pred_instances.append(real_pred_this)

    return pred_instances, output_instances, real_pred_instances


def setup(rank, world_size):
    """
    :param rank: rank of current thread
    :param world_size: the total number of threads
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    clean up the opened threads
    """
    dist.destroy_process_group()


def run_function(func, world_size):
    """
    :param func: the function to call in each thread
    :param world_size: the world size of the distributed system
    """
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def run_correlation(func, world_size, dataset_idx):
    """
    :param func: function to call in each thread
    :param world_size: world size of the distributed system
    :param dataset_idx: dataset idx of this layer learning
    """
    mp.spawn(func,
             args=(world_size, dataset_idx, ),
             nprocs=world_size,
             join=True)


def crop_single_instance(img, centroid, target_length=64):
    """
    :param img: original img
    :param centroid: centroid of the instance
    :param target_length: target size of the cropped instance
    :return: cropped instance
    """
    height, width = img.shape[1:]

    start_x, start_y, end_x, end_y = [centroid[1]-target_length//2, centroid[0]-target_length//2,
                      centroid[1]+target_length//2, centroid[0]+target_length//2]

    delta_height, delta_width = end_y - start_y, end_x - start_x
    res = torch.zeros((img.shape[0], delta_height, delta_width))
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

    res[:, dsy:dey, dsx:dex] = img[:, start_y:end_y, start_x:end_x]

    if padding_bool[0]:
        res[:, :, :dsx] = (res[:, :, dsx:dsx * 2]).flip(2)

    if padding_bool[1]:
        res[:, :, dex:] = (res[:, :, dex - (delta_width - dex):dex]).flip(2)

    if padding_bool[2]:
        res[:, :dsy, :] = (res[:, dsy:dsy * 2:, :]).flip(1)

    if padding_bool[3]:
        res[:, dey:, :] = (res[:, dey - (delta_height - dey):dey, :]).flip(1)

    return res


def crop_img_instance_with_bg(img, mask, target_width=64):
    """
    :param img: original img
    :param mask: mask of the instance
    :param target_width: target size of the cropped patch
    :return: cropped patch
    """
    indexes = torch.nonzero((mask > 0.99), as_tuple=False)

    x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()
    y_min, y_max = torch.min(indexes[:, 2]).item(), torch.max(indexes[:, 2]).item()

    centroid = [(x_max + x_min) // 2, (y_max + y_min) // 2]

    if (x_max - x_min < target_width) and (y_max - y_min < target_width):
        instance_now = crop_single_instance(img, centroid, target_length=target_width)
        return instance_now

    max_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
    instance_now = crop_single_instance(img, centroid, target_length=max_length)
    instance_now = TF.resize(instance_now, (target_width, target_width), \
                             interpolation=PIL.Image.NEAREST)
    return instance_now
