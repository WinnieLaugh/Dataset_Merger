"""
Training script for sparse training, with only loss of the ground truth labels
"""
import os
import sys
sys.path.append('./')

from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50
from dataset.dataset import LinearEvalMultiFCDatasetCrop

from model.utils import loss_fn, get_accuracy, setup, \
                        cleanup, run_function, dataset_mask, get_pred_true_f1

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--name", type=str, default="dataset_merging", help="name of experiment")
parser.add_argument("--max_epoch", type=int, default=50, help="max epoch number")
parser.add_argument("--root_dir", type=str,
                    default="/data",
                    help="the data folder where data sets are stored")

args = parser.parse_args()


def sparse_training(rank, world_size):
    """
    the base function to train with sparse labelling
    :param rank: rank of the thread
    :param world_size: total number of threads
    """
    setup(rank, world_size)

    dataset_names = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]

    imagenet = LinearEvalMultiFCDatasetCrop(args.root_dir, dataset_names, split_name="train_v1")
    imagenet_test = LinearEvalMultiFCDatasetCrop(args.root_dir, \
                                                 dataset_names, split_name="valid_v1")

    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=args.batch_size,
                                                      num_workers=8, sampler=sampler)

    sampler_test = DistributedSampler(imagenet_test)
    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test, \
                                                           batch_size=args.batch_size, \
                                                            num_workers=8, sampler=sampler_test)

    resnet = resnet50(num_classes=35)
    resnet = resnet.to(rank)

    if rank == 0:
        writer = SummaryWriter("logs/{}/sparse_training".format(args.name))
        print("len of training set: ", len(imagenet))
        print("len of validate set: ", len(imagenet_test))

    ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet)
    ddp_model = DDP(ddp_model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(ddp_model.module.parameters(), lr=args.lr, weight_decay=1e-4)

    dist.barrier()

    for _ in range(args.max_epoch):
        ddp_model.train()
        loss_sum = 0
        sampler.set_epoch(_)

        for image, dataset_idxes, type_idx in imagenet_dataloader:
            image = image.to(rank).float()
            dataset_idxes = dataset_idxes.to(rank)
            type_idx = type_idx.to(rank)
            output = ddp_model(image)
            loss = loss_fn(output, dataset_idxes, type_idx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.module.parameters(), 10)
            optimizer.step()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()

        if rank == 0:
            print("epoch ", _, "loss", loss_sum / len(imagenet))
            writer.add_scalar('Loss/loss', loss_sum / len(imagenet), _)

            # check point save
            if _ % 10 == 0 or (_ == (args.max_epoch - 1)):
                os.makedirs("checkpoints/{}/sparse_training".format(args.name), exist_ok=True)
                checkpoint_path = "checkpoints/{}/sparse_training/improved-net_{}.pt".format(
                                                                                args.name, _)
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                checkpoint_path = "checkpoints/{}/sparse_training/improved-net_latest.pt".format(
                                                                                        args.name)
                torch.save(ddp_model.module.state_dict(), checkpoint_path)

        if _ % 10 == 0:
            with torch.no_grad():
                ddp_model.eval()
                total_num = torch.zeros(len(dataset_mask)).to(rank)
                correct_array = torch.zeros(len(dataset_mask)).to(rank)
                for image, dataset_idxes, type_idx in imagenet_test_dataloader:
                    image = image.to(rank).float()
                    dataset_idxes = dataset_idxes.to(rank)
                    type_idx = type_idx.to(rank)
                    output = resnet(image)

                    correct_num_this, total_num_this = get_accuracy(output, dataset_idxes, type_idx)
                    correct_array += correct_num_this
                    total_num += total_num_this

                if rank == 0:
                    for dataset_idx, dataset_name in enumerate(dataset_names):
                        print("accuracy ", dataset_name, " ",
                              correct_array[dataset_idx] / total_num[dataset_idx])
                    print("accuracy ", correct_array.sum() / total_num.sum())

        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()


def linear_eval():
    """
    This function is to evaluate the model trained with sparse labels
    """
    device = torch.device(0)

    dataset_names = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]
    imagenet_test = LinearEvalMultiFCDatasetCrop(args.root_dir, dataset_names, split_name="test_v1")

    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test, \
                                                           batch_size=args.batch_size, \
                                                           num_workers=8, shuffle=True)
    resnet = resnet50(num_classes=35)
    resnet = resnet.to(device)

    checkpoint_path = "checkpoints/{}/sparse_training/improved-net_latest.pt".format(args.name)
    resnet.load_state_dict(torch.load(checkpoint_path))
    with torch.no_grad():
        resnet.eval()
        correct_array = torch.zeros(len(dataset_names))
        total_num = torch.zeros(len(dataset_names))
        y_pred = [[], [], [], [], []]
        y_true = [[], [], [], [], []]
        for image, dataset_idxes, type_idx in imagenet_test_dataloader:
            image = image.to(device).float()
            dataset_idxes = dataset_idxes.to(device)
            type_idx = type_idx.to(device)
            output = resnet(image)

            correct_num_this, total_num_this, y_pred_this, y_true_this = \
                        get_pred_true_f1(output, dataset_idxes, type_idx)
            correct_array += correct_num_this
            total_num += total_num_this

            for dataset_idx, _ in enumerate(dataset_mask):
                y_pred[dataset_idx] += y_pred_this[dataset_idx]
                y_true[dataset_idx] += y_true_this[dataset_idx]

        for dataset_idx, dataset_name in enumerate(dataset_names):
            y_pred_this = np.array(y_pred[dataset_idx])
            y_true_this = np.array(y_true[dataset_idx])

            print(dataset_name,
                  " accuracy: ",   correct_array[dataset_idx] / total_num[dataset_idx],
                   " f1: ",       f1_score(y_pred_this, y_true_this, average='weighted'),
                  " total num: ", total_num[dataset_idx])


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_function(sparse_training, n_gpus)
    linear_eval()
