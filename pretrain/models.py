# BSD 3-Clause License
#
# Copyright (c) 2022-23
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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


def remove_inplace(model):
    for name, module in model.named_children():
        # Replace ReLU
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=False))
        # Replace Dropout while preserving its parameters
        elif isinstance(module, torch.nn.Dropout):
            dropout_params = module.state_dict()
            new_dropout = torch.nn.Dropout(**dropout_params, inplace=False)
            setattr(model, name, new_dropout)
        else:
            # Recursively apply the replacements to child modules
            remove_inplace(module)


class SeriesModel(torch.nn.Module):
    def __init__(self):
        super(SeriesModel, self).__init__()
        self.attn_linear2 = torch.nn.Linear(D2, 1)

    def forward(self, x):
        (batch, series, channels, height, width) = x.shape
        shape = [-1, channels, height, width]
        x = x.reshape(*shape)  # (batch * series, channels, height, width)
        x = self.net(x).squeeze()  # (batch * series, E)

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
    def __init__(self, pretrained: bool = True, channels: int = CH):
        super(SeriesEfficientNetb0, self).__init__()

        # EfficientNet b0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        remove_inplace(net)
        self.net = torch.nn.Sequential(
            net.features,
            net.avgpool,
        )

        # Change number of input channels
        net.features[0][0] = torch.nn.Conv2d(
            channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # Classifier (for attention) and attention
        self.classifier = net.classifier
        D1 = self.classifier[-1].out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)

        # Embedding dimensions
        self.embedding_dim = self.classifier[1].in_features


class SeriesMobileNetv3(SeriesModel):
    def __init__(self, pretrained: bool = True, channels: int = CH):
        super(SeriesMobileNetv3, self).__init__()

        # MobileNet
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        remove_inplace(net)
        self.net = torch.nn.Sequential(
            net.features,
            net.avgpool,
        )

        # Change number of input channels
        net.features[0][0] = torch.nn.Conv2d(
            channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # Classifier (for attention) and attention
        self.classifier = net.classifier
        D1 = self.classifier[-1].out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)

        # Embedding dimensions
        self.embedding_dim = self.classifier[0].in_features


class SeriesResNet18(SeriesModel):
    def __init__(self, pretrained: bool = True, channels: int = CH):
        super(SeriesResNet18, self).__init__()

        # ResNet 18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)
        remove_inplace(self.net)

        # Change number of input channels
        self.net.conv1 = torch.nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Classifier (for attention) and attention
        self.classifier = self.net.fc
        self.net.fc = torch.nn.Identity()
        D1 = self.classifier.out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)

        # Embedding dimensions
        self.embedding_dim = self.classifier.in_features


class SeriesResNet34(SeriesModel):
    def __init__(self, pretrained: bool = True, channels: int = CH):
        super(SeriesResNet34, self).__init__()

        # ResNet 34
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.net = models.resnet34(weights=weights)
        remove_inplace(self.net)

        # Change number of input channels
        self.net.conv1 = torch.nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Classifier (for attention) and attention
        self.classifier = self.net.fc
        self.net.fc = torch.nn.Identity()
        D1 = self.classifier.out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)

        # Embedding dimensions
        self.embedding_dim = self.classifier.in_features


class SeriesResNet50(SeriesModel):
    def __init__(self, pretrained: bool = True, channels: int = CH):
        super(SeriesResNet50, self).__init__()

        # ResNet 50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.net = models.resnet50(weights=weights)
        remove_inplace(self.net)

        # Change number of input channels
        self.net.conv1 = torch.nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Classifier (for attention) and attention
        self.classifier = self.net.fc
        self.net.fc = torch.nn.Identity()
        D1 = self.classifier.out_features
        self.attn_linear1 = torch.nn.Linear(D1, D2)

        # Embedding dimensions
        self.embedding_dim = self.classifier.in_features
