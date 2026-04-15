"""Compact AlexNet-style model for CIFAR-10 (SiLU activations).

The original AlexNet uses max-pooling and large 224x224 inputs. For Rotom e2e,
we keep the AlexNet pattern (stacked conv stages then MLP head) but use stride-2
convs for downsampling on 32x32 CIFAR inputs.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class AlexNetSmall(nn.Module):
    """AlexNet-style conv backbone + two-layer classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.SiLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.SiLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()

        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.SiLU()

        # Global pool to [batch, 128], then two-layer MLP head.
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.fc1_act = nn.SiLU()

    def forward_features(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = F.avg_pool2d(x, x.size(-1))
        return x.view(x.size(0), -1)

    def forward_features_through_stage2(self, x):
        """Prefix through conv2/bn2/act2."""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

    def forward_features_through_stage3(self, x):
        """Prefix through conv3/bn3/act3 (matches ``depth == "stage3"`` in TensorIR)."""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc1_act(self.fc1(x))
        x = self.fc2(x)
        return x


def alexnet_small(num_classes: int = 10) -> AlexNetSmall:
    return AlexNetSmall(num_classes=num_classes)
