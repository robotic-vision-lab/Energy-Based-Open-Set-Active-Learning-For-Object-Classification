"""Active learning loop for EB-OSAL.

This module owns cycle-level orchestration:
- build per-cycle dataloaders
- train EKUS
- mine pseudo-unknowns
- train ESS
- run the query strategy
- update the pool
- evaluate and log per-cycle results

It does not own backbone definitions, dataset construction, or low-level model
components. Those stay in their respective modules.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
# from torch.utils.data import DataLoader

from data import (
    DataBundle,
    PoolState,
    build_labeled_loader,
    build_test_loader,
    build_unlabeled_loader,
    summarize_data_state,
    update_pool_state,
)
from methods.ekus import (
    EKUSConfig,
    EKUSModel,
    evaluate_ekus_known_unknown_separation,
    mine_pseudo_unknown_indices,
    train_ekus_one_epoch,
)
from methods.ess import ESSConfig, ESSModel, evaluate_ess, train_ess_one_epoch
from methods.query_strategies import QueryResult, run_query_strategy


# -----------------------------------------------------------------------------
# Records
# -----------------------------------------------------------------------------


@dataclass
class CycleRecord:
    cycle: int
    num_labeled: int
    num_unlabeled: int
    ekus_train: Dict[str, float] = field(default_factory=dict)
    ekus_eval: Dict[str, float] = field(default_factory=dict)
    ess_train: Dict[str, float] = field(default_factory=dict)
    ess_eval: Dict[str, float] = field(default_factory=dict)
    query: Dict[str, Any] = field(default_factory=dict)
    data_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLoopResult:
    history: List[CycleRecord]
    final_pool_state: PoolState


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _get_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg["name"].lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {cfg['name']}")



def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epochs: int,
):
    name = cfg.get("name", "none").lower()

    if name in {"none", "null", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(cfg.get("min_lr", 1e-6)),
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfg.get("milestones", [100, 150])),
            gamma=float(cfg.get("gamma", 0.1)),
        )

    raise ValueError(f"Unsupported scheduler: {cfg.get('name')}")


def _step_scheduler(scheduler) -> None:
    if scheduler is not None:
        scheduler.step()


def _save_checkpoint(
    save_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    cycle: int,
    stage_name: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{stage_name}_cycle_{cycle:02d}.pt"
    payload = {
        "cycle": cycle,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _record_train_output(obj) -> Dict[str, float]:
    if hasattr(obj, "__dict__"):
        return {k: float(v) for k, v in obj.__dict__.items()}
    return {}


# -----------------------------------------------------------------------------
# Stage training wrappers
# -----------------------------------------------------------------------------
def train_ekus_stage(
    model: EKUSModel,
    data_bundle: DataBundle,
    pool_state: PoolState,
    optimizer_cfg: Dict[str, Any],
    scheduler_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    ekus_cfg: EKUSConfig,
    batch_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, float], List[int]]:
    """Train EKUS for one AL cycle and return pseudo-unknown indices."""
    epochs = int(training_cfg["epochs"])
    batch_size = int(batch_cfg["train"])
    num_workers = int(training_cfg.get("num_workers", 4))

    # labeled_loader = build_labeled_loader(
    #     data_bundle.train_dataset,
    #     pool_state,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )
    """Modify to receive data_bundle.split.known_class_to_label"""
    labeled_loader = build_labeled_loader(
        data_bundle.train_dataset,
        pool_state,
        data_bundle.split.known_class_to_label,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    unlabeled_loader = build_unlabeled_loader(
        data_bundle.train_dataset,
        pool_state,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    optimizer = _get_optimizer(model, optimizer_cfg)
    scheduler = _get_scheduler(optimizer, scheduler_cfg, epochs)

    pseudo_unknown_indices: List[int] = []
    final_train_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        # Mine pseudo-unknowns from the current model state. On the first epoch,
        # this will reflect the randomly initialized or previously restored EKUS.
        pseudo_unknown_indices = mine_pseudo_unknown_indices(
            model=model,
            unlabeled_loader=unlabeled_loader,
            pseudo_unknown_ratio=ekus_cfg.pseudo_unknown_ratio,
            device=device,
        )

        train_out = train_ekus_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            device=device,
            config=ekus_cfg,
            pseudo_unknown_indices=pseudo_unknown_indices,
        )
        final_train_metrics = _record_train_output(train_out)
        _step_scheduler(scheduler)

    final_train_metrics["num_pseudo_unknown"] = float(len(pseudo_unknown_indices))
    return final_train_metrics, pseudo_unknown_indices



def train_ess_stage(
    model: ESSModel,
    data_bundle: DataBundle,
    pool_state: PoolState,
    optimizer_cfg: Dict[str, Any],
    scheduler_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    ess_cfg: ESSConfig,
    batch_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Train ESS for one AL cycle."""
    epochs = int(training_cfg["epochs"])
    batch_size = int(batch_cfg["train"])
    num_workers = int(training_cfg.get("num_workers", 4))

    # labeled_loader = build_labeled_loader(
    #     data_bundle.train_dataset,
    #     pool_state,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )
    labeled_loader = build_labeled_loader(
        data_bundle.train_dataset,
        pool_state,
        data_bundle.split.known_class_to_label,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer = _get_optimizer(model, optimizer_cfg)
    scheduler = _get_scheduler(optimizer, scheduler_cfg, epochs)

    final_train_metrics: Dict[str, float] = {}
    for epoch in range(epochs):
        train_out = train_ess_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            optimizer=optimizer,
            device=device,
            config=ess_cfg,
        )
        final_train_metrics = _record_train_output(train_out)
        _step_scheduler(scheduler)

    return final_train_metrics


# -----------------------------------------------------------------------------
# Query + evaluation wrappers
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_cycle(
    ekus_model: EKUSModel,
    ess_model: ESSModel,
    data_bundle: DataBundle,
    pool_state: PoolState,
    batch_cfg: Dict[str, Any],
    known_classes: List[int],
    device: torch.device,
    num_workers: int = 4,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate EKUS separation and ESS known-class accuracy for the cycle."""
    test_batch_size = int(batch_cfg["test"])

    # full_test_loader = build_test_loader(
    #     data_bundle.test_dataset,
    #     data_bundle.split,
    #     batch_size=test_batch_size,
    #     num_workers=num_workers,
    #     known_only=False,
    # )
    # known_test_loader = build_test_loader(
    #     data_bundle.test_dataset,
    #     data_bundle.split,
    #     batch_size=test_batch_size,
    #     num_workers=num_workers,
    #     known_only=True,
    # )
    """
    full test loader keeps original labels for known/unknown separation
    known-only test loader uses remapped labels for ESS classification evaluation
    """
    full_test_loader = build_test_loader(
        data_bundle.test_dataset,
        data_bundle.split,
        batch_size=test_batch_size,
        num_workers=num_workers,
        known_only=False,
        remap_known_labels=False,
    )
    known_test_loader = build_test_loader(
        data_bundle.test_dataset,
        data_bundle.split,
        batch_size=test_batch_size,
        num_workers=num_workers,
        known_only=True,
        remap_known_labels=True,
    )

    ekus_eval = evaluate_ekus_known_unknown_separation(
        model=ekus_model,
        data_loader=full_test_loader,
        known_classes=known_classes,
        device=device,
    )
    ess_eval = evaluate_ess(
        model=ess_model,
        data_loader=known_test_loader,
        device=device,
    )
    return ekus_eval, ess_eval


def run_query_stage(
    strategy_name: str,
    ekus_model: EKUSModel,
    ess_model: ESSModel,
    data_bundle: DataBundle,
    pool_state: PoolState,
    active_learning_cfg: Dict[str, Any],
    ekus_cfg: EKUSConfig,
    ess_cfg: ESSConfig,
    seed: int,
    batch_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> QueryResult:
    """Run the requested query strategy for the current cycle."""
    return run_query_strategy(
        strategy_name=strategy_name,
        pool_state=pool_state,
        train_dataset=data_bundle.train_dataset,
        budget=int(active_learning_cfg["query_budget"]),
        seed=seed,
        batch_size=int(batch_cfg["query"]),
        num_workers=int(training_cfg.get("num_workers", 4)),
        device=device,
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        ekus_model=ekus_model,
        ess_model=ess_model,
        ekus_threshold=(
            float(ekus_cfg.threshold)
            if ekus_cfg.threshold is not None
            else float(ekus_cfg.energy_margin_known)
        ),
        ess_score_weight=float(ess_cfg.score_weight),
    )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run_active_learning_loop(
    ekus_model: EKUSModel,
    ess_model: ESSModel,
    data_bundle: DataBundle,
    config: Dict[str, Any],
    device: torch.device,
    outputs_dir: Optional[str | Path] = None,
) -> ActiveLoopResult:
    """Run the full EB-OSAL active learning loop.

    Expected config structure:
        seed
        active_learning
        training
        optimizer
        scheduler
        batch_size
        method:
            ekus
            ess

    Notes:
    - EKUS and ESS are trained from their current state each cycle.
    - If you want to reinitialize every cycle instead, that policy should be
      implemented at a higher level or by extending this function.
    """
    seed = int(config["seed"])
    active_learning_cfg = config["active_learning"]
    training_cfg = dict(config["training"])
    optimizer_cfg = config["optimizer"]
    scheduler_cfg = config["scheduler"]
    batch_cfg = config["batch_size"]
    ekus_cfg = EKUSConfig.from_dict(config["method"]["ekus"])
    ess_cfg = ESSConfig.from_dict(config["method"]["ess"])

    # Allow top-level defaults to flow into stage builders without duplicating
    # too much config logic elsewhere.
    training_cfg.setdefault("num_workers", int(config.get("num_workers", 4)))
    training_cfg.setdefault("pin_memory", bool(config.get("pin_memory", True)))

    num_cycles = int(active_learning_cfg["num_cycles"])
    strategy_name = str(active_learning_cfg["query_strategy"])

    outputs_path = Path(outputs_dir) if outputs_dir is not None else None
    checkpoints_dir = outputs_path / "checkpoints" if outputs_path is not None else None
    logs_dir = outputs_path / "logs" if outputs_path is not None else None
    results_dir = outputs_path / "results" if outputs_path is not None else None

    history: List[CycleRecord] = []
    pool_state = copy.deepcopy(data_bundle.pool_state)

    ekus_model = ekus_model.to(device)
    ess_model = ess_model.to(device)

    for cycle in range(num_cycles):
        current_bundle = DataBundle(
            dataset_name=data_bundle.dataset_name,
            train_dataset=data_bundle.train_dataset,
            test_dataset=data_bundle.test_dataset,
            num_classes=data_bundle.num_classes,
            split=data_bundle.split,
            pool_state=pool_state,
        )

        # 1. Train EKUS.
        ekus_train_metrics, pseudo_unknown_indices = train_ekus_stage(
            model=ekus_model,
            data_bundle=current_bundle,
            pool_state=pool_state,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            training_cfg=training_cfg,
            ekus_cfg=ekus_cfg,
            batch_cfg=batch_cfg,
            device=device,
        )

        # 2. Train ESS.
        ess_train_metrics = train_ess_stage(
            model=ess_model,
            data_bundle=current_bundle,
            pool_state=pool_state,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            training_cfg=training_cfg,
            ess_cfg=ess_cfg,
            batch_cfg=batch_cfg,
            device=device,
        )

        # 3. Evaluate current state before querying the next batch.
        ekus_eval_metrics, ess_eval_metrics = evaluate_cycle(
            ekus_model=ekus_model,
            ess_model=ess_model,
            data_bundle=current_bundle,
            pool_state=pool_state,
            batch_cfg=batch_cfg,
            known_classes=current_bundle.split.known_classes,
            device=device,
            num_workers=int(training_cfg.get("num_workers", 4)),
        )

        # 4. Query new samples.
        query_result = run_query_stage(
            strategy_name=strategy_name,
            ekus_model=ekus_model,
            ess_model=ess_model,
            data_bundle=current_bundle,
            pool_state=pool_state,
            active_learning_cfg=active_learning_cfg,
            ekus_cfg=ekus_cfg,
            ess_cfg=ess_cfg,
            seed=seed + cycle,
            batch_cfg=batch_cfg,
            training_cfg=training_cfg,
            device=device,
        )

        # 5. Record the cycle before updating the pool, so the data state shown
        # here corresponds to the state the models actually trained on.
        cycle_record = CycleRecord(
            cycle=cycle,
            num_labeled=len(pool_state.labeled_indices),
            num_unlabeled=len(pool_state.unlabeled_indices),
            ekus_train={
                **ekus_train_metrics,
                "num_pseudo_unknown": float(len(pseudo_unknown_indices)),
            },
            ekus_eval=ekus_eval_metrics,
            ess_train=ess_train_metrics,
            ess_eval=ess_eval_metrics,
            query={
                "strategy": query_result.strategy_name,
                "selected_indices": query_result.selected_indices,
                "num_selected": len(query_result.selected_indices),
                **query_result.metadata,
            },
            data_state=summarize_data_state(current_bundle),
        )
        history.append(cycle_record)

        # 6. Save per-cycle artifacts if requested.
        if checkpoints_dir is not None:
            _save_checkpoint(checkpoints_dir, ekus_model, None, cycle, "ekus")
            _save_checkpoint(checkpoints_dir, ess_model, None, cycle, "ess")

        if logs_dir is not None:
            _write_json(logs_dir / f"cycle_{cycle:02d}.json", asdict(cycle_record))

        # 7. Update the labeled/unlabeled pool for the next cycle.
        pool_state = update_pool_state(pool_state, query_result.selected_indices)

    result = ActiveLoopResult(history=history, final_pool_state=pool_state)

    if results_dir is not None:
        payload = {
            "history": [asdict(record) for record in history],
            "final_pool_state": {
                "num_labeled": len(result.final_pool_state.labeled_indices),
                "num_unlabeled": len(result.final_pool_state.unlabeled_indices),
                "queried_history": result.final_pool_state.queried_history,
            },
        }
        _write_json(results_dir / "summary.json", payload)

    return result