"""EKUS: Energy-based Known/Unknown Separator.

This module owns:
- the EKUS model wrapper
- energy computation
- EKUS training losses
- pseudo-unknown mining from the unlabeled pool
- EKUS train / eval utilities

The implementation is intentionally compact and keeps the loss definitions close
to the method logic, since they are not reused broadly elsewhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Energy helpers
# -----------------------------------------------------------------------------
def compute_energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute free energy from class logits.

    E(x) = -log sum_y exp(f_y(x))

    Lower energy indicates stronger compatibility with known-class structure.
    """
    return -torch.logsumexp(logits, dim=1)


def sample_random_complementary_labels(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Sample a random complementary label for each sample.

    For each ground-truth label y, sample y_bar != y.
    This is mainly useful when applying negative learning to labeled data.
    For unlabeled data, complementary labels can be sampled uniformly from all
    known classes and this function is not required.
    """
    if num_classes < 2:
        raise ValueError("Need at least 2 classes to sample complementary labels.")

    random_offsets = torch.randint(
        low=1,
        high=num_classes,
        size=labels.shape,
        device=labels.device,
    )
    return (labels + random_offsets) % num_classes


def sample_uniform_labels(
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample labels uniformly from the known class set."""
    return torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def ekus_hinge_loss(
    known_energy: torch.Tensor,
    unknown_energy: torch.Tensor,
    margin_known: float,
    margin_unknown: float,
) -> torch.Tensor:
    """Margin-based separation loss for EKUS.

    Known samples are encouraged to stay below margin_known.
    Pseudo-unknown samples are encouraged to stay above margin_unknown.
    """
    known_term = F.relu(known_energy - margin_known).pow(2)
    unknown_term = F.relu(margin_unknown - unknown_energy).pow(2)
    return known_term.mean() + unknown_term.mean()


def ekus_contrastive_loss(
    known_energy: torch.Tensor,
    unknown_energy: torch.Tensor,
) -> torch.Tensor:
    """Simple energy separation term.

    Minimizing mean(E_known - E_unknown) pushes known samples lower and pseudo-
    unknown samples higher in energy.
    """
    return (known_energy.mean() - unknown_energy.mean())


def negative_learning_loss(
    logits: torch.Tensor,
    complementary_labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative-learning regularization.

    For each sample, discourage confident assignment to a sampled label.
    This matches the high-level role described in the revised paper, where NL is
    used to prevent unlabeled samples from being absorbed too aggressively into
    known classes.
    """
    probs = F.softmax(logits, dim=1)
    selected = probs.gather(1, complementary_labels.view(-1, 1)).squeeze(1)
    return (-torch.log(1.0 - selected.clamp(max=1.0 - eps) + eps)).mean()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class EKUSModel(nn.Module):
    """EKUS model.

    The backbone is expected to expose either:
    - forward(x, return_features=True) -> (logits, features)
    - forward_features(x) -> features

    EKUS attaches its own classifier head so it can be trained and scored
    independently from ESS even when they share the same underlying architecture.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, tuple):
                return out[0]
            return out

        out = self.backbone(x)
        if isinstance(out, tuple):
            return out[-1]
        return out

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_energy: bool = False,
    ):
        features = self.forward_features(x)
        logits = self.classifier(features)

        if return_features and return_energy:
            energy = compute_energy_from_logits(logits)
            return logits, features, energy
        if return_features:
            return logits, features
        if return_energy:
            energy = compute_energy_from_logits(logits)
            return logits, energy
        return logits

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return compute_energy_from_logits(logits)


# -----------------------------------------------------------------------------
# Config / outputs
# -----------------------------------------------------------------------------
@dataclass
class EKUSConfig:
    energy_margin_known: float
    energy_margin_unknown: float
    contrastive_weight: float = 0.2
    negative_learning_weight: float = 0.2
    pseudo_unknown_ratio: float = 0.05
    threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "EKUSConfig":
        return cls(
            energy_margin_known=float(cfg["energy_margin_known"]),
            energy_margin_unknown=float(cfg["energy_margin_unknown"]),
            contrastive_weight=float(cfg.get("contrastive_weight", 0.2)),
            negative_learning_weight=float(cfg.get("negative_learning_weight", 0.2)),
            pseudo_unknown_ratio=float(cfg.get("pseudo_unknown_ratio", 0.05)),
            threshold=cfg.get("threshold"),
        )


@dataclass
class EKUSTrainOutput:
    loss: float
    hinge_loss: float
    contrastive_loss: float
    negative_learning_loss: float

# -----------------------------------------------------------------------------
# Pseudo-unknown mining
# -----------------------------------------------------------------------------
@torch.no_grad()
def score_unlabeled_energy(
    model: EKUSModel,
    unlabeled_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    """Score unlabeled samples with EKUS energy.

    Returns:
        indices: original dataset indices
        energies: energy values aligned with `indices`
    """
    model.eval()
    all_indices: List[int] = []
    all_energies: List[float] = []

    for batch in unlabeled_loader:
        inputs, _, indices = batch
        inputs = inputs.to(device, non_blocking=True)
        energy = model.energy(inputs)

        all_indices.extend([int(i) for i in indices])
        all_energies.extend(energy.detach().cpu().tolist())

    return all_indices, all_energies


@torch.no_grad()
def mine_pseudo_unknown_indices(
    model: EKUSModel,
    unlabeled_loader: DataLoader,
    pseudo_unknown_ratio: float,
    device: torch.device,
) -> List[int]:
    """Select the top-rho highest-energy unlabeled samples as pseudo-unknowns."""
    indices, energies = score_unlabeled_energy(model, unlabeled_loader, device)
    if len(indices) == 0:
        return []

    num_select = max(1, int(round(len(indices) * pseudo_unknown_ratio)))
    ranked = sorted(zip(indices, energies), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:num_select]]


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
def _to_device(batch, device: torch.device):
    inputs, labels, indices = batch
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return inputs, labels, indices


# def _build_pseudo_unknown_batch(
#     unlabeled_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
#     pseudo_unknown_index_set: set[int],
# ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#     """Extract pseudo-unknown samples from an unlabeled batch.
#
#     The dataloader returns original dataset indices, which lets us select only
#     the samples mined as pseudo-unknowns in the current AL cycle.
#     """
#     inputs, labels, indices = unlabeled_batch
#     keep_mask = torch.tensor(
#         [int(idx) in pseudo_unknown_index_set for idx in indices],
#         dtype=torch.bool,
#         device=inputs.device,
#     )
#
#     if keep_mask.sum().item() == 0:
#         return None
#
#     return inputs[keep_mask], labels[keep_mask], indices[keep_mask]


def _build_pseudo_unknown_batch(
    inputs: torch.Tensor,
    indices,
    pseudo_unknown_index_set: set[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if torch.is_tensor(indices):
        keep_mask_cpu = torch.tensor(
            [int(idx) in pseudo_unknown_index_set for idx in indices],
            dtype=torch.bool,
            device=indices.device,
        )
    else:
        keep_mask_cpu = torch.tensor(
            [int(idx) in pseudo_unknown_index_set for idx in indices],
            dtype=torch.bool,
        )

    if keep_mask_cpu.sum().item() == 0:
        return None

    keep_mask_inputs = keep_mask_cpu.to(inputs.device)
    return inputs[keep_mask_inputs], indices[keep_mask_cpu]


def train_ekus_one_epoch(
    model: EKUSModel,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: EKUSConfig,
    pseudo_unknown_indices: Optional[Sequence[int]] = None,
) -> EKUSTrainOutput:
    """Train EKUS for one epoch.

    Training uses:
    - supervised CE on labeled known samples
    - hinge + contrastive energy separation between labeled knowns and mined
      pseudo-unknowns
    - negative learning on the full unlabeled batch

    If no pseudo-unknown samples are present in a minibatch, the separation terms
    for that minibatch fall back to zero.
    """
    model.train()

    ce_meter = 0.0
    hinge_meter = 0.0
    contrastive_meter = 0.0
    nl_meter = 0.0
    total_meter = 0.0
    num_steps = 0

    pseudo_unknown_index_set = set(pseudo_unknown_indices or [])
    unlabeled_iter = iter(unlabeled_loader)

    for labeled_batch in labeled_loader:
        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)

        labeled_inputs, labeled_targets, _ = _to_device(labeled_batch, device)
        unlabeled_inputs, _, unlabeled_indices = _to_device(unlabeled_batch, device)

        optimizer.zero_grad()

        labeled_logits = model.forward_logits(labeled_inputs)
        labeled_energy = compute_energy_from_logits(labeled_logits)
        ce_loss = F.cross_entropy(labeled_logits, labeled_targets)

        # Negative learning on all unlabeled samples.
        unlabeled_logits = model.forward_logits(unlabeled_inputs)
        sampled_comp_labels = sample_uniform_labels(
            batch_size=unlabeled_logits.size(0),
            num_classes=model.num_classes,
            device=unlabeled_logits.device,
        )
        nl_loss = negative_learning_loss(unlabeled_logits, sampled_comp_labels)

        # Separation losses on pseudo-unknown subset, if available.
        # pseudo_unknown_batch = _build_pseudo_unknown_batch(
        #     (unlabeled_inputs, torch.zeros_like(unlabeled_indices, device=unlabeled_inputs.device), unlabeled_indices),
        #     pseudo_unknown_index_set,
        # )
        pseudo_unknown_batch = _build_pseudo_unknown_batch(
            unlabeled_inputs,
            unlabeled_indices,
            pseudo_unknown_index_set,
        )

        if pseudo_unknown_batch is not None:
            pseudo_inputs = pseudo_unknown_batch[0]
            pseudo_logits = model.forward_logits(pseudo_inputs)
            pseudo_energy = compute_energy_from_logits(pseudo_logits)
            hinge_loss = ekus_hinge_loss(
                known_energy=labeled_energy,
                unknown_energy=pseudo_energy,
                margin_known=config.energy_margin_known,
                margin_unknown=config.energy_margin_unknown,
            )
            contrastive_loss = ekus_contrastive_loss(labeled_energy, pseudo_energy)
        else:
            hinge_loss = torch.zeros((), device=device)
            contrastive_loss = torch.zeros((), device=device)

        loss = (
            ce_loss
            + hinge_loss
            + config.contrastive_weight * contrastive_loss
            + config.negative_learning_weight * nl_loss
        )
        loss.backward()
        optimizer.step()

        total_meter += float(loss.item())
        ce_meter += float(ce_loss.item())
        hinge_meter += float(hinge_loss.item())
        contrastive_meter += float(contrastive_loss.item())
        nl_meter += float(nl_loss.item())
        num_steps += 1

    if num_steps == 0:
        return EKUSTrainOutput(0.0, 0.0, 0.0, 0.0)

    return EKUSTrainOutput(
        loss=total_meter / num_steps,
        hinge_loss=hinge_meter / num_steps,
        contrastive_loss=contrastive_meter / num_steps,
        negative_learning_loss=nl_meter / num_steps,
    )


@torch.no_grad()
def evaluate_ekus_known_unknown_separation(
    model: EKUSModel,
    data_loader: DataLoader,
    known_classes: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate simple threshold-free separation statistics on a loader.

    This function assumes the loader returns global dataset indices but uses the
    provided batch labels to compute known-vs-unknown partitioning.
    """
    model.eval()
    known_set = set(int(c) for c in known_classes)

    known_energy_values: List[float] = []
    unknown_energy_values: List[float] = []

    for batch in data_loader:
        inputs, labels, _ = batch
        inputs = inputs.to(device, non_blocking=True)
        energy = model.energy(inputs).detach().cpu()

        for e, y in zip(energy.tolist(), labels.tolist()):
            if int(y) in known_set:
                known_energy_values.append(float(e))
            else:
                unknown_energy_values.append(float(e))

    result = {}
    if known_energy_values:
        result["known_energy_mean"] = sum(known_energy_values) / len(known_energy_values)
    else:
        result["known_energy_mean"] = 0.0

    if unknown_energy_values:
        result["unknown_energy_mean"] = sum(unknown_energy_values) / len(unknown_energy_values)
    else:
        result["unknown_energy_mean"] = 0.0

    result["energy_gap"] = result["unknown_energy_mean"] - result["known_energy_mean"]
    return result


@torch.no_grad()
def filter_likely_known_samples(
    model: EKUSModel,
    unlabeled_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[List[int], List[int], Dict[int, float]]:
    """Filter unlabeled samples by EKUS energy threshold.

    Returns:
        likely_known_indices
        likely_unknown_indices
        energy_by_index
    """
    indices, energies = score_unlabeled_energy(model, unlabeled_loader, device)

    likely_known: List[int] = []
    likely_unknown: List[int] = []
    energy_by_index: Dict[int, float] = {}

    for idx, energy in zip(indices, energies):
        energy_by_index[idx] = float(energy)
        if energy < threshold:
            likely_known.append(idx)
        else:
            likely_unknown.append(idx)

    return likely_known, likely_unknown, energy_by_index

# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_ekus(
    backbone: nn.Module,
    feature_dim: int,
    num_classes: int,
) -> EKUSModel:
    return EKUSModel(backbone=backbone, feature_dim=feature_dim, num_classes=num_classes)