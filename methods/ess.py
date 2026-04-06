"""ESS: Energy-based Sample Scorer.

This module owns:
- the ESS model wrapper
- ESS energy computation
- ESS training losses
- uncertainty / entropy computation
- ESS scoring utilities for active learning

ESS operates only on samples that are treated as likely known after EKUS
filtering. Its role is not known/unknown separation, but ranking those retained
samples by informativeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import SubsetWithIndex


# -----------------------------------------------------------------------------
# Energy / uncertainty helpers
# -----------------------------------------------------------------------------



def compute_energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute free energy from class logits.

    Lower energy indicates stronger compatibility with one of the known classes.
    """
    return -torch.logsumexp(logits, dim=1)



def compute_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute predictive entropy from class logits.

    Higher entropy indicates greater predictive uncertainty.
    """
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)



def compute_max_softmax_probability(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------



def ess_energy_regularization_loss(
    energy: torch.Tensor,
    margin_known: float,
) -> torch.Tensor:
    """Encourage confidently learned known samples to stay in low-energy regions."""
    return F.relu(energy - margin_known).pow(2).mean()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ESSModel(nn.Module):
    """ESS model.

    ESS attaches its own classifier head on top of the shared backbone. This
    keeps its training state independent from EKUS even if both use the same
    architecture family.
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
        return_entropy: bool = False,
    ):
        features = self.forward_features(x)
        logits = self.classifier(features)

        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_energy:
            outputs.append(compute_energy_from_logits(logits))
        if return_entropy:
            outputs.append(compute_entropy_from_logits(logits))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return compute_energy_from_logits(logits)

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return compute_entropy_from_logits(logits)


# -----------------------------------------------------------------------------
# Config / outputs
# -----------------------------------------------------------------------------


@dataclass
class ESSConfig:
    energy_margin_known: float
    energy_reg_weight: float = 0.1
    score_weight: float = 0.1

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ESSConfig":
        return cls(
            energy_margin_known=float(cfg["energy_margin_known"]),
            energy_reg_weight=float(cfg.get("energy_reg_weight", 0.1)),
            score_weight=float(cfg.get("score_weight", 0.1)),
        )


@dataclass
class ESSTrainOutput:
    loss: float
    ce_loss: float
    energy_reg_loss: float
    accuracy: float


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------



def _to_device(batch, device: torch.device):
    inputs, labels, indices = batch
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return inputs, labels, indices


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------



def train_ess_one_epoch(
    model: ESSModel,
    labeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ESSConfig,
) -> ESSTrainOutput:
    """Train ESS for one epoch on labeled known-class data."""
    model.train()

    total_loss_meter = 0.0
    ce_meter = 0.0
    reg_meter = 0.0
    correct = 0
    total = 0
    num_steps = 0

    for batch in labeled_loader:
        inputs, targets, _ = _to_device(batch, device)

        optimizer.zero_grad()

        logits = model.forward_logits(inputs)
        energy = compute_energy_from_logits(logits)

        ce_loss = F.cross_entropy(logits, targets)
        reg_loss = ess_energy_regularization_loss(
            energy=energy,
            margin_known=config.energy_margin_known,
        )
        loss = ce_loss + config.energy_reg_weight * reg_loss

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.numel())

        total_loss_meter += float(loss.item())
        ce_meter += float(ce_loss.item())
        reg_meter += float(reg_loss.item())
        num_steps += 1

    if num_steps == 0:
        return ESSTrainOutput(loss=0.0, ce_loss=0.0, energy_reg_loss=0.0, accuracy=0.0)

    accuracy = float(correct) / max(total, 1)
    return ESSTrainOutput(
        loss=total_loss_meter / num_steps,
        ce_loss=ce_meter / num_steps,
        energy_reg_loss=reg_meter / num_steps,
        accuracy=accuracy,
    )


@torch.no_grad()
def evaluate_ess(
    model: ESSModel,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ESS as a standard classifier on a labeled or known-only test set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_energy = 0.0
    total_entropy = 0.0
    num_steps = 0

    for batch in data_loader:
        inputs, targets, _ = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model.forward_logits(inputs)
        energy = compute_energy_from_logits(logits)
        entropy = compute_entropy_from_logits(logits)
        loss = F.cross_entropy(logits, targets)

        preds = logits.argmax(dim=1)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.numel())

        total_loss += float(loss.item())
        total_energy += float(energy.mean().item())
        total_entropy += float(entropy.mean().item())
        num_steps += 1

    if num_steps == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "mean_energy": 0.0,
            "mean_entropy": 0.0,
        }

    return {
        "loss": total_loss / num_steps,
        "accuracy": float(total_correct) / max(total_count, 1),
        "mean_energy": total_energy / num_steps,
        "mean_entropy": total_entropy / num_steps,
    }


# -----------------------------------------------------------------------------
# Query-time scoring
# -----------------------------------------------------------------------------


@dataclass
class ESSScoreOutput:
    indices: List[int]
    scores: List[float]
    entropy: List[float]
    energy: List[float]
    msp: List[float]


@torch.no_grad()
def score_candidate_loader(
    model: ESSModel,
    data_loader: DataLoader,
    device: torch.device,
    score_weight: float,
) -> ESSScoreOutput:
    """Score a candidate set using entropy + beta * energy.

    This matches the scoring function in the paper.
    """
    model.eval()

    all_indices: List[int] = []
    all_scores: List[float] = []
    all_entropy: List[float] = []
    all_energy: List[float] = []
    all_msp: List[float] = []

    for batch in data_loader:
        inputs, _, indices = batch
        inputs = inputs.to(device, non_blocking=True)

        logits = model.forward_logits(inputs)
        entropy = compute_entropy_from_logits(logits)
        energy = compute_energy_from_logits(logits)
        msp = compute_max_softmax_probability(logits)
        score = entropy + score_weight * energy

        all_indices.extend([int(i) for i in indices])
        all_scores.extend(score.detach().cpu().tolist())
        all_entropy.extend(entropy.detach().cpu().tolist())
        all_energy.extend(energy.detach().cpu().tolist())
        all_msp.extend(msp.detach().cpu().tolist())

    return ESSScoreOutput(
        indices=all_indices,
        scores=all_scores,
        entropy=all_entropy,
        energy=all_energy,
        msp=all_msp,
    )


@torch.no_grad()
def rank_candidate_indices(
    model: ESSModel,
    dataset: Dataset,
    candidate_indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    score_weight: float,
    pin_memory: bool = True,
) -> ESSScoreOutput:
    """Score and rank a subset of candidate indices.

    This is the standard entry point used after EKUS filters the unlabeled pool.
    """
    if len(candidate_indices) == 0:
        return ESSScoreOutput(indices=[], scores=[], entropy=[], energy=[], msp=[])

    subset = SubsetWithIndex(dataset, candidate_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return score_candidate_loader(
        model=model,
        data_loader=loader,
        device=device,
        score_weight=score_weight,
    )


@torch.no_grad()
def select_top_k_candidates(
    score_output: ESSScoreOutput,
    budget: int,
) -> List[int]:
    """Return the top-k indices according to ESS score."""
    if budget <= 0 or len(score_output.indices) == 0:
        return []

    ranked = sorted(
        zip(score_output.indices, score_output.scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [idx for idx, _ in ranked[:budget]]


@torch.no_grad()
def build_score_dict(score_output: ESSScoreOutput) -> Dict[int, Dict[str, float]]:
    """Build per-index score metadata for logging or analysis."""
    result: Dict[int, Dict[str, float]] = {}
    for idx, score, ent, energy, msp in zip(
        score_output.indices,
        score_output.scores,
        score_output.entropy,
        score_output.energy,
        score_output.msp,
    ):
        result[int(idx)] = {
            "score": float(score),
            "entropy": float(ent),
            "energy": float(energy),
            "msp": float(msp),
        }
    return result


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------



def build_ess(
    backbone: nn.Module,
    feature_dim: int,
    num_classes: int,
) -> ESSModel:
    return ESSModel(backbone=backbone, feature_dim=feature_dim, num_classes=num_classes)
