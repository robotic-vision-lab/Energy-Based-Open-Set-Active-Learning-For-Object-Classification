"""PointNet backbone for 3D point-cloud classification in EB-OSAL.

This implementation follows the standard PointNet classification design with a
T-Net for input alignment, shared MLP layers, global max pooling, and a final
classification head. It also exposes global feature embeddings for EKUS / ESS.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.fc3.weight, 0.0)
        identity = torch.eye(self.k).view(-1)
        with torch.no_grad():
            self.fc3.bias.copy_(identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, k, N]
        batch_size = x.size(0)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = torch.max(out, dim=2, keepdim=False)[0]

        out = F.relu(self.bn4(self.fc1(out)), inplace=True)
        out = F.relu(self.bn5(self.fc2(out)), inplace=True)
        out = self.fc3(out)
        out = out.view(batch_size, self.k, self.k)
        return out


class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform: bool = True) -> None:
        super().__init__()
        self.feature_transform = feature_transform

        self.input_transform = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        if self.feature_transform:
            self.feature_transform_net = TNet(k=64)
        else:
            self.feature_transform_net = None

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_dim = 1024

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Input x expected as [B, N, 3]
        if x.dim() != 3:
            raise ValueError(f"Expected point cloud tensor of shape [B, N, 3], got {tuple(x.shape)}")
        if x.size(-1) != 3:
            raise ValueError(f"Expected last dimension to be 3, got {x.size(-1)}")

        x = x.transpose(1, 2)  # [B, 3, N]

        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        feature_transform = None
        if self.feature_transform and self.feature_transform_net is not None:
            feature_transform = self.feature_transform_net(x)
            x = torch.bmm(feature_transform, x)

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        global_features = torch.max(x, dim=2, keepdim=False)[0]  # [B, 1024]

        return global_features, input_transform, feature_transform


class PointNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_transform: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(feature_transform=feature_transform)
        self.feature_dim = self.encoder.feature_dim

        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.encoder(x)

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(features)), inplace=True)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))), inplace=True)
        logits = self.fc3(x)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_transforms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, input_transform, feature_transform = self.forward_features(x)
        logits = self.forward_head(features)

        if return_transforms:
            return logits, features, input_transform, feature_transform
        if return_features:
            return logits, features
        return logits

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        features, input_transform, feature_transform = self.forward_features(x)
        logits = self.forward_head(features)
        return {
            "logits": logits,
            "features": features,
            "input_transform": input_transform,
            "feature_transform": feature_transform,
        }


class PointNetBackbone(nn.Module):
    """Backbone wrapper exposing only the feature extractor interface."""

    def __init__(self, feature_transform: bool = True) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feature_transform=feature_transform)
        self.feature_dim = self.encoder.feature_dim

    def forward(
        self,
        x: torch.Tensor,
        return_transforms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, input_transform, feature_transform = self.encoder(x)
        if return_transforms:
            return features, input_transform, feature_transform
        return features



def feature_transform_regularizer(transform: torch.Tensor) -> torch.Tensor:
    """Regularization term used in the original PointNet.

    Encourages the learned feature transform to stay close to orthogonal.
    """
    if transform.dim() != 3:
        raise ValueError(f"Expected transform shape [B, K, K], got {tuple(transform.shape)}")

    batch_size, k, _ = transform.shape
    identity = torch.eye(k, device=transform.device).unsqueeze(0).expand(batch_size, -1, -1)
    diff = torch.bmm(transform, transform.transpose(2, 1)) - identity
    return torch.mean(torch.norm(diff, dim=(1, 2)))



def pointnet_cls(
    num_classes: int,
    feature_transform: bool = True,
    dropout: float = 0.3,
) -> PointNetClassifier:
    return PointNetClassifier(
        num_classes=num_classes,
        feature_transform=feature_transform,
        dropout=dropout,
    )