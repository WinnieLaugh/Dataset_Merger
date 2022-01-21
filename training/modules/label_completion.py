"""
Script to train label completion module. We complete pseudo labels for the missing labels in the
multi-label ground truth sets.
"""
import os
import sys
sys.path.append('./')

from argparse import ArgumentParser
import shutil
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50

from dataset.dataset import LinearEvalMultiFCSupUnsupDatasetCrop, LinearEvalMultiFCDatasetCrop
from model.utils import setup, cleanup, run_function, dataset_mask
from model.utils import get_accuracy_label_completion, loss_fn_label_completion
from model.model import MultiLabelModelLabelCompletion



parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--name", type=str, default="dataset_merging", help="name of experiment")
parser.add_argument("--max_epoch", type=int, default=50, help="max epoch number")
parser.add_argument("--threshold", type=float, default=0.99, help="threshold of the confidence")
parser.add_argument("--root_dir", type=str,
                    default="/data",
                    help="the data folder where data sets are stored")

args = parser.parse_args()


def label_completion_training(rank, world_size):
    """
    Function to train the label completion module
    :param rank: the rank of the thread
    :param world_size: the total number of threads
    """
    setup(rank, world_size)

    dataset_names_supervised = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]
    dataset_names_unsupervised = ['CryoNuSeg', 'cpm15', 'cpm17', 'kumar', 'MoNuSeg', 'tnbc']

    imagenet = LinearEvalMultiFCSupUnsupDatasetCrop(
        args.root_dir,
        dataset_names_sup=dataset_names_supervised,
        dataset_names_unsup=dataset_names_unsupervised,
        split_name="train_v1")
    imagenet_test = LinearEvalMultiFCSupUnsupDatasetCrop(
        args.root_dir,
        dataset_names_sup=dataset_names_supervised,
        dataset_names_unsup=dataset_names_unsupervised,
        split_name="valid_v1")

    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=args.batch_size,
                                                      num_workers=8, sampler=sampler)

    sampler_test = DistributedSampler(imagenet_test)
    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test,
                                                           batch_size=args.batch_size,
                                                           num_workers=8, sampler=sampler_test)

    if rank == 0:
        if os.path.exists("logs/{}/label_completion".format(args.name)):
            shutil.rmtree("logs/{}/label_completion".format(args.name))
        writer = SummaryWriter("logs/{}/label_completion".format(args.name))
        print(args)

    resnet = resnet50(num_classes=35)
    resnet_checkpoint_path =\
        "checkpoints/{}/sparse_training/improved-net_latest.pt".format(args.name)
    resnet.load_state_dict(torch.load(resnet_checkpoint_path))
    resnet = resnet.to(rank)

    classifiers = [nn.Sequential(nn.Linear(35 - len(dataset_mask[_]), 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, len(dataset_mask[_])))
                                            for _ in range(len(dataset_mask))]

    for dataset_idx in range(len(dataset_mask)):
        checkpoint_path = \
            "checkpoints/{}/label_correlation/{}/improved-net_latest.pt".format(args.name,
                                                    dataset_names_supervised[dataset_idx])
        classifiers[dataset_idx].load_state_dict(torch.load(checkpoint_path))
        classifiers[dataset_idx] = classifiers[dataset_idx].to(rank)
        classifiers[dataset_idx] = \
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifiers[dataset_idx])

    model = MultiLabelModelLabelCompletion(resnet=resnet, classifiers=classifiers)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizers = torch.optim.Adam(ddp_model.module.parameters(), lr=args.lr, weight_decay=1e-4)

    dist.barrier()

    for _ in range(args.max_epoch):
        ddp_model.train()

        loss_sum = 0
        loss_0_sum = torch.zeros((len(dataset_mask), 2)).to(rank)
        loss_1_sum = torch.zeros((len(dataset_mask), 2), dtype=torch.float).to(rank)
        used_length_sum = torch.zeros((len(dataset_mask), 2), dtype=torch.int).to(rank)

        sampler.set_epoch(_)

        for image, dataset_idxes, type_idxes in imagenet_dataloader:
            image = image.to(rank).float()
            type_idxes = type_idxes.to(rank)
            dataset_idxes = dataset_idxes.to(rank)

            pred, out = ddp_model(image, False)
            losses, loss_0, loss_1, used_length = \
                loss_fn_label_completion(pred, out, dataset_idxes, type_idxes, args.threshold)

            optimizers.zero_grad()

            losses.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.module.parameters(), 10)
            optimizers.step()

            if rank == 0:
                for dataset_idx in range(len(dataset_mask)):
                    loss_sum += losses
                    loss_0_sum += loss_0
                    loss_1_sum += loss_1
                    used_length_sum += used_length

        if rank == 0:
            print("loss cross entropy pred: ", loss_0_sum[dataset_idx, 0].detach().cpu().numpy())
            print("===============epoch ", _, "=========================")

        # check point save
        if rank == 0:
            if _ % 10 == 0 or (_ == args.max_epoch - 1):
                os.makedirs("checkpoints/{}/label_completion".format(args.name), exist_ok=True)
                checkpoint_path = \
                    "checkpoints/{}/label_completion/improved-net_{}.pt".format(args.name, _)
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                checkpoint_path = \
                    "checkpoints/{}/label_completion/improved-net_latest.pt".format(args.name)
                torch.save(ddp_model.module.state_dict(), checkpoint_path)

        if _ % 10 == 0 or (_ == args.max_epoch - 1):
            with torch.no_grad():
                ddp_model.eval()

                correct_array = torch.zeros((len(dataset_mask), 3))
                total_num = torch.zeros((len(dataset_mask), ))
                for image, dataset_idxes, type_idxes in imagenet_test_dataloader:
                    image = image.to(rank).float()
                    dataset_idxes = dataset_idxes.to(rank)
                    type_idxes = type_idxes.to(rank)

                    pred, out = ddp_model(image)

                    correct_num_this, total_num_this =\
                        get_accuracy_label_completion(pred, out, dataset_idxes, type_idxes)

                    correct_array += correct_num_this
                    total_num += total_num_this

                if rank == 0:
                    for dataset_idx in range(len(dataset_mask)):
                        print(dataset_names_supervised[dataset_idx],
                        " pred: ", correct_array[dataset_idx, 0] / total_num[dataset_idx],
                        " output: ", correct_array[dataset_idx, 1] / total_num[dataset_idx],
                        " final: ", correct_array[dataset_idx, 1] / total_num[dataset_idx])
                        writer.add_scalars(f'accuracy/{dataset_names_supervised[dataset_idx]}',
                        {'pred': correct_array[dataset_idx, 0] / total_num[dataset_idx],
                        'ouput': correct_array[dataset_idx, 1] / total_num[dataset_idx],
                        'final': correct_array[dataset_idx, 2] / total_num[dataset_idx]}, _)

        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()


def linear_eval():
    """
    Function to evaluate the label completion module
    """
    device = torch.device(0)

    dataset_names_supervised = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]

    imagenet_test = LinearEvalMultiFCDatasetCrop(args.root_dir,
                                                 dataset_names=dataset_names_supervised,
                                                 split_name="test_v1")

    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test,
                                                           batch_size=args.batch_size,
                                                           num_workers=8, shuffle=True)

    resnet = resnet50(num_classes=35)
    classifiers = [nn.Sequential(nn.Linear(35 - len(dataset_mask[_]), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, len(dataset_mask[_]))) for _ in range(len(dataset_mask))]
    model = MultiLabelModelLabelCompletion(resnet=resnet, classifiers=classifiers)

    checkpoint_path = "checkpoints/{}/label_completion/improved-net_latest.pt".format(args.name)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        correct_array = torch.zeros((len(dataset_mask), 3))
        total_num = torch.zeros((len(dataset_mask),))
        pred_instances = [[] for _ in range(len(dataset_mask))]
        output_instances = [[] for _ in range(len(dataset_mask))]
        real_this = [[] for _ in range(len(dataset_mask))]
        read_pred_instances = [[] for _ in range(len(dataset_mask))]

        for image, dataset_idxes, type_idxes in imagenet_test_dataloader:
            image = image.to(device).float()
            dataset_idxes = dataset_idxes.to(device)
            type_idxes = type_idxes.to(device)

            pred, out = model(image)

            correct_num_this, total_num_this,\
            pred_instances_this, output_instances_this, real_pred_instances_this, real_this_this \
                        = get_accuracy_label_completion(pred, out, dataset_idxes,
                                                        type_idxes, f1_calculation=True)

            correct_array += correct_num_this
            total_num += total_num_this

            for dataset_idx in range(len(dataset_mask)):
                pred_instances[dataset_idx] += pred_instances_this[dataset_idx]
                output_instances[dataset_idx] += output_instances_this[dataset_idx]
                read_pred_instances[dataset_idx] += real_pred_instances_this[dataset_idx]
                real_this[dataset_idx] += real_this_this[dataset_idx]

        for dataset_idx in range(len(dataset_mask)):
            y_pred_this = np.array(pred_instances[dataset_idx])
            y_output_this = np.array(output_instances[dataset_idx])
            y_true_this = np.array(real_this[dataset_idx])
            read_pred_this = np.array(read_pred_instances[dataset_idx])

            print("accuracy pred: ", dataset_names_supervised[dataset_idx],
                  " ", correct_array[dataset_idx, 0] / total_num[dataset_idx],
                  " output: ", correct_array[dataset_idx, 1] / total_num[dataset_idx],
                  "mergr: ", correct_array[dataset_idx, 2] / total_num[dataset_idx],
                  " f1 pred: ", f1_score(y_pred_this, y_true_this, average='weighted'),
                  " f1 output: ", f1_score(y_output_this, y_true_this, average='weighted'),
                  " f1 merge: ", f1_score(read_pred_this, y_true_this, average='weighted'),
                  " total num: ", total_num[dataset_idx])


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_function(label_completion_training, n_gpus)
    print("#################################################")
    linear_eval()
