import copy

import torch
import torch.nn.functional as F
from torchvision import models

CH = 12
D2 = 256


def freeze(m: torch.nn.Module) -> torch.nn.Module:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m: torch.nn.Module) -> torch.nn.Module:
    for p in m.parameters():
        p.requires_grad = True


class SeriesModel(torch.nn.Module):

    def __init__(self):
        super(SeriesModel, self).__init__()
        self.attn_linear2 = torch.nn.Linear(D2, 1)

    def forward(self, x):
        (batch, series, channels, height, width) = x.shape
        shape = [-1, channels, height, width]
        x = x.reshape(*shape)  # (batch * series, channels, height, width)
        x = self.net(x)  # (batch * series, E)

        attn_weights = self.classifier(x)  # (batch * series, D1)
        attn_weights = self.attn_linear1(attn_weights)  # (batch * series, D2)
        attn_weights = F.relu(attn_weights)
        attn_weights = self.attn_linear2(attn_weights)  # (batch * series, 1)
        shape = [batch, series, 1]
        attn_weights = attn_weights.reshape(*shape)  # (batch, series, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        shape = list(x.shape)
        shape = [batch, series] + shape[1:]
        x = x.reshape(*shape)  # (batch, series, E)
        x = x * attn_weights
        x = torch.sum(x, dim=1)  # (batch, E)
        return x


class SeriesEfficientNetb0(SeriesModel):

    def __init__(self, pretrained: bool = True):
        super(SeriesEfficientNetb0, self).__init__()
        self.net = models.efficientnet_b0(pretrained=pretrained)
        self.classifier = self.net.classifier
        self.net.classifier = torch.nn.Identity()
        self.net.features[0][0] = torch.nn.Conv2d(CH,
                                                  32,
                                                  kernel_size=(3, 3),
                                                  stride=(2, 2),
                                                  padding=(1, 1),
                                                  bias=False)
        D1 = self.classifier[-1].out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)


class SeriesMobileNetv3(SeriesModel):

    def __init__(self, pretrained: bool = True):
        super(SeriesMobileNetv3, self).__init__()
        self.net = models.mobilenet_v3_small(pretrained=pretrained)
        self.classifier = self.net.classifier
        self.net.classifier = torch.nn.Identity()
        self.net.features[0][0] = torch.nn.Conv2d(CH,
                                                  16,
                                                  kernel_size=(3, 3),
                                                  stride=(2, 2),
                                                  padding=(1, 1),
                                                  bias=False)

        D1 = self.classifier[-1].out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)


class SeriesResNet18(SeriesModel):

    def __init__(self, pretrained: bool = True):
        super(SeriesResNet18, self).__init__()
        self.net = models.resnet18(pretrained=pretrained)
        self.classifier = self.net.fc
        self.net.fc = torch.nn.Identity()
        self.net.conv1 = torch.nn.Conv2d(CH,
                                         64,
                                         kernel_size=(7, 7),
                                         stride=(2, 2),
                                         padding=(3, 3),
                                         bias=False)

        D1 = self.classifier.out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)
