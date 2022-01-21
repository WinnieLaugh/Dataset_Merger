"""
Scipt to learn label correlation layers. We train five two-layer fully connected layers
to learn correlation among different sets of labels
"""
import os
import sys
sys.path.append('./')

from sklearn.metrics import f1_score
import numpy as np
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torchvision.models.resnet import resnet50
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import LinearEvalSingleDatasetCrop
from model.utils import loss_fn_correlation_learning, get_accuracy_correlation_learning, \
                        dataset_mask, get_other_probabilities
from model.utils import run_correlation, setup, cleanup

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--name", type=str, default="dataset_merging", help="name of experiment")
parser.add_argument("--max_epoch", type=int, default=50, help="max epoch number")
parser.add_argument("--dataset_idx", type=int, default=0, help="the dataset used now")
parser.add_argument("--num_workers", type=int, default=8, help="the dataset used now")
parser.add_argument("--root_dir", type=str,
                    default="/data",
                    help="the data folder where data sets are stored")

args = parser.parse_args()


def label_correlation_learning(rank, world_size, dataset_idx):
    """
    function to learn label correlation layers
    :param rank: the rank of this thread
    :param world_size: total number of threads
    :param dataset_idx: the dataset index to process now
    """
    setup(rank, world_size)

    dataset_names_supervised = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]

    imagenet = LinearEvalSingleDatasetCrop(args.root_dir, dataset_names=dataset_names_supervised,
                                           dataset_idx=dataset_idx, split_name="train_v1")
    imagenet_test = LinearEvalSingleDatasetCrop(
            args.root_dir, dataset_names=dataset_names_supervised,
            dataset_idx=dataset_idx, split_name="valid_v1")
    resnet_checkpoint_path = \
        "checkpoints/{}/sparse_training/improved-net_latest.pt".format(args.name)

    print("checkpoint path: ", resnet_checkpoint_path)
    sampler = DistributedSampler(imagenet)
    imagenet_dataloader = torch.utils.data.DataLoader(dataset=imagenet, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, sampler=sampler)

    sampler_test = DistributedSampler(imagenet_test)
    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test, \
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           sampler=sampler_test)

    if rank == 0:
        print(args)
        writer = SummaryWriter("logs/{}/label_correlation/{}".format(args.name,
                                        dataset_names_supervised[dataset_idx]))
        print("check on ", dataset_names_supervised[dataset_idx])

    resnet = resnet50(num_classes=35)
    resnet.load_state_dict(torch.load(resnet_checkpoint_path))
    resnet = resnet.to(rank)
    resnet.eval()

    classifier = nn.Sequential(nn.Linear(35 - len(dataset_mask[dataset_idx]), 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, len(dataset_mask[dataset_idx])))
    ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    ddp_model = ddp_model.to(rank)
    ddp_model = DDP(ddp_model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(ddp_model.module.parameters(), lr=args.lr, weight_decay=1e-4)

    dist.barrier()

    for _ in range(args.max_epoch):
        ddp_model.train()
        loss_sum = 0

        sampler.set_epoch(_)

        for image, type_idxes in imagenet_dataloader:
            image = image.to(rank).float()
            type_idxes = type_idxes.to(rank)

            with torch.no_grad():
                pred = resnet(image).detach()
                out = get_other_probabilities(pred, dataset_idx)

            out = ddp_model(out)
            loss = loss_fn_correlation_learning(out, type_idxes)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if rank == 0:
                loss_sum += loss.detach().cpu().numpy()

        if rank == 0:
            print("epoch ", _, "loss", loss_sum / len(imagenet))
            writer.add_scalar('loss_total', loss_sum / len(imagenet), _)

            if _ % 10 == 0 or (_ == args.max_epoch - 1):
                os.makedirs("checkpoints/{}/label_correlation/{}".format(args.name,
                                            dataset_names_supervised[dataset_idx]),
                                                                    exist_ok=True)
                checkpoint_path = "checkpoints/{}/label_correlation/{}/improved-net_{}.pt".format(
                                                                                        args.name,
                                                            dataset_names_supervised[dataset_idx],
                                                                                                _)
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                checkpoint_path = \
                    "checkpoints/{}/label_correlation/{}/improved-net_latest.pt".format(
                                                                               args.name,
                                                    dataset_names_supervised[dataset_idx])
                torch.save(ddp_model.module.state_dict(), checkpoint_path)

        if _ % 10 == 0:
            with torch.no_grad():
                ddp_model.eval()
                correct_array = torch.zeros(3)
                total_num = 0
                for image, type_idxes in imagenet_test_dataloader:
                    image = image.to(rank).float()
                    type_idxes = type_idxes.to(rank)

                    pred = resnet(image)
                    out = get_other_probabilities(pred, dataset_idx)
                    out = ddp_model(out)
                    indexes = range(image.shape[0])
                    pred = pred[indexes,
                           dataset_mask[dataset_idx][0]:(dataset_mask[dataset_idx][-1]+1)]

                    correct_num_this, total_num_this = get_accuracy_correlation_learning(pred,
                                                                                         out,
                                                                                         type_idxes)

                    correct_array += correct_num_this
                    total_num += total_num_this

                if rank == 0:
                    print("pred results accuracy: ")
                    print(dataset_names_supervised[dataset_idx],
                            " pred: ", correct_array[0] / total_num,
                            " output: ", correct_array[1] / total_num,
                            " mean: ", correct_array[2] / total_num)
                    writer.add_scalars(f'accuracy/{dataset_names_supervised[dataset_idx]}',
                                        {'pred': correct_array[0] / total_num,
                                        'ouput': correct_array[1] / total_num},
                                           _)

        dist.barrier()

    if rank == 0:
        writer.close()

    cleanup()


def linear_eval(dataset_idx):
    """
    Evaluate the learned correlation layers
    :param dataset_idx: the dataset idx to test label correlation of all other sets
    """
    device = torch.device(0)

    dataset_names = ["NuCLS", "BreCaHAD", "CoNSeP", "MoNuSAC", "panNuke"]

    imagenet_test = LinearEvalSingleDatasetCrop(args.root_dir,
            dataset_names=dataset_names, dataset_idx=dataset_idx, split_name="test_v1")
    resnet_checkpoint_path = \
        "checkpoints/{}/sparse_training/improved-net_latest.pt".format(args.name)

    imagenet_test_dataloader = torch.utils.data.DataLoader(dataset=imagenet_test,
                                                           batch_size=args.batch_size,
                                                           num_workers=8, shuffle=True)
    resnet = resnet50(num_classes=35)
    resnet.load_state_dict(torch.load(resnet_checkpoint_path))
    resnet = resnet.to(device)

    checkpoint_path = \
        "checkpoints/{}/label_correlation/{}/improved-net_latest.pt".format(args.name,
                                                            dataset_names[dataset_idx])
    classifier = nn.Sequential(nn.Linear(35 - len(dataset_mask[dataset_idx]), 128),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Linear(128, len(dataset_mask[dataset_idx])))
    classifier.load_state_dict(torch.load(checkpoint_path))
    classifier = classifier.to(device)

    with torch.no_grad():
        resnet.eval()
        classifier.eval()
        correct_array = torch.zeros(3)
        total_num = 0
        pred_total = []
        output_total = []
        real_total = []
        type_this_total = []

        for image, type_idxes in imagenet_test_dataloader:
            image = image.to(device).float()
            type_idxes = type_idxes.to(device)

            pred = resnet(image)
            out = get_other_probabilities(pred, dataset_idx)
            out = classifier(out)
            indexes = range(image.shape[0])
            pred = pred[indexes, dataset_mask[dataset_idx][0]:(dataset_mask[dataset_idx][-1] + 1)]

            correct_num_this, total_num_this, pred_this, output_this, real_this, type_idxes = \
                                    get_accuracy_correlation_learning(pred, out, type_idxes, True)

            correct_array += correct_num_this
            total_num += total_num_this
            pred_total += pred_this
            output_total += output_this
            real_total += real_this
            type_this_total += type_idxes

        pred_total = np.array(pred_total)
        output_total = np.array(output_total)
        real_total = np.array(real_total)
        type_this_total = np.array(type_this_total)
        print("accuracy ", dataset_names[dataset_idx], " pred: ", correct_array[0] / total_num,
                                                    " output: ", correct_array[1] / total_num,
                                                    " real: ", correct_array[2] / total_num,
                " f1 output: ", f1_score(output_total, type_this_total, average='weighted'),
                " f1 pred: ", f1_score(pred_total, type_this_total, average='weighted'),
                " f1 real: ", f1_score(real_total, type_this_total, average='weighted'))


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    for i in range(len(dataset_mask)):
        run_correlation(label_correlation_learning, n_gpus, i)
        linear_eval(i)
