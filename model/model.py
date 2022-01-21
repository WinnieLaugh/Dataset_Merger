"""
The model for label completion module.
"""
import torch
import torch.nn as nn
from model.utils import dataset_mask, get_multiple_probabilities


class MultiLabelModelLabelCompletion(nn.Module):
    """
    The model for label completion module, with resnet50 and label correlation layers
    """
    def __init__(self, resnet, classifiers, istrain=True):
        super().__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.istrain = istrain

        self.fc = resnet.fc

        for dataset_idx in range(len(dataset_mask)):
            setattr(self, 'classifier_{}'.format(dataset_idx), classifiers[dataset_idx])

    def forward(self, x, detach=True):
        """
        :param x: input data
        :param detach: where to detach the predictions from resnet50 or not
        :return: x_pred: predictions from the resnet50
                 x_out: predictions from the label correlation layers
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_pred = self.fc(x)
        if detach:
            x_out = self.get_out(x_pred.detach())
        else:
            x_out = self.get_out(x_pred)

        return x_pred, x_out

    def get_out(self, x):
        """
        :param x: input data
        :return: predictions of label correlation layers
        """
        x_out = []
        x = get_multiple_probabilities(x)
        for dataset_idx in range(len(dataset_mask)):
            classifier = getattr(self, 'classifier_{}'.format(dataset_idx))
            x_out.append(classifier(x[dataset_idx]))

        return x_out
