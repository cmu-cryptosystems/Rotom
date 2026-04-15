"""Compact MobileNet-style CIFAR model (SiLU, affine BN).

Blocks use a **3×3 expand** conv followed by a **1×1 project** conv (standard convolutions
only), matching common inverted-bottleneck *channel* patterns without depthwise grouping
so the graph stays in the existing ``TensorTerm.conv2d`` subset.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class MobileNetSmall(nn.Module):
    """Stem + three MB-style blocks + global average pool + linear classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_stem = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_stem = nn.BatchNorm2d(16)
        self.act_stem = nn.SiLU()

        self.conv_b1_exp = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn_b1_exp = nn.BatchNorm2d(32)
        self.act_b1_exp = nn.SiLU()
        self.conv_b1_pw = nn.Conv2d(
            32, 16, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_b1_pw = nn.BatchNorm2d(16)
        self.act_b1_pw = nn.SiLU()

        self.conv_b2_exp = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn_b2_exp = nn.BatchNorm2d(32)
        self.act_b2_exp = nn.SiLU()
        self.conv_b2_pw = nn.Conv2d(
            32, 24, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_b2_pw = nn.BatchNorm2d(24)
        self.act_b2_pw = nn.SiLU()

        self.conv_b3_exp = nn.Conv2d(
            24, 48, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b3_exp = nn.BatchNorm2d(48)
        self.act_b3_exp = nn.SiLU()
        self.conv_b3_pw = nn.Conv2d(
            48, 32, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_b3_pw = nn.BatchNorm2d(32)
        self.act_b3_pw = nn.SiLU()

        self.fc = nn.Linear(32, num_classes)

    def forward_features_through_stage2(self, x):
        """Prefix through block1 (matches TensorIR ``depth="stage2"``)."""
        x = self.act_stem(self.bn_stem(self.conv_stem(x)))
        x = self.act_b1_exp(self.bn_b1_exp(self.conv_b1_exp(x)))
        x = self.act_b1_pw(self.bn_b1_pw(self.conv_b1_pw(x)))
        return x

    def forward_features_through_stage3(self, x):
        """Prefix through block2 (matches TensorIR ``depth="stage3"``)."""
        x = self.forward_features_through_stage2(x)
        x = self.act_b2_exp(self.bn_b2_exp(self.conv_b2_exp(x)))
        x = self.act_b2_pw(self.bn_b2_pw(self.conv_b2_pw(x)))
        return x

    def forward_features(self, x):
        x = self.forward_features_through_stage3(x)
        x = self.act_b3_exp(self.bn_b3_exp(self.conv_b3_exp(x)))
        x = self.act_b3_pw(self.bn_b3_pw(self.conv_b3_pw(x)))
        x = F.avg_pool2d(x, x.size(-1))
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(x)


def mobilenet_small(num_classes: int = 10) -> MobileNetSmall:
    return MobileNetSmall(num_classes=num_classes)
