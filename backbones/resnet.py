"""ResNet backbone for 2D image classification in EB-OSAL.

This module provides a lightweight ResNet implementation tailored for CIFAR-style
inputs. It exposes both class logits and intermediate feature embeddings so EKUS
and ESS can reuse the same backbone cleanly.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    """CIFAR-style ResNet backbone.

    This version uses a 3x3 stem instead of the ImageNet-style 7x7 + maxpool,
    which is more appropriate for 32x32 inputs.
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: list[int],
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.num_classes = num_classes
        self.feature_dim = base_channels * 8 * block.expansion

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.forward_head(features)
        if return_features:
            return logits, features
        return logits

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.forward_head(features)
        return {
            "logits": logits,
            "features": features,
        }


class ResNetBackbone(nn.Module):
    """Wrapper exposing a unified interface for downstream method modules.

    EKUS and ESS can share this backbone and optionally attach separate heads.
    """

    def __init__(self, backbone: ResNet) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.num_classes = backbone.num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x, return_features=return_features)



def build_resnet(
    depth: int,
    num_classes: int,
    in_channels: int = 3,
    base_channels: int = 64,
    dropout: float = 0.0,
) -> ResNet:
    """Factory for common ResNet variants.

    Supported depths:
        18, 34, 50
    """
    if depth == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels, base_channels, dropout)
    if depth == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels, base_channels, dropout)
    if depth == 50:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels, base_channels, dropout)
    raise ValueError(f"Unsupported ResNet depth: {depth}")



def resnet18(num_classes: int, in_channels: int = 3, dropout: float = 0.0) -> ResNet:
    return build_resnet(18, num_classes=num_classes, in_channels=in_channels, dropout=dropout)



def resnet34(num_classes: int, in_channels: int = 3, dropout: float = 0.0) -> ResNet:
    return build_resnet(34, num_classes=num_classes, in_channels=in_channels, dropout=dropout)



def resnet50(num_classes: int, in_channels: int = 3, dropout: float = 0.0) -> ResNet:
    return build_resnet(50, num_classes=num_classes, in_channels=in_channels, dropout=dropout)

