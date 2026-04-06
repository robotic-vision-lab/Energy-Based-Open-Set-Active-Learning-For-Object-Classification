"""Query strategies for EB-OSAL.

This module owns only sample-selection policies. It does not train EKUS or ESS,
and it does not update the pool state. Its job is to take already-trained models,
score the current unlabeled pool, and return queried indices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from data import PoolState, SubsetWithIndex, build_loader
from methods.ekus import EKUSModel, filter_likely_known_samples
from methods.ess import (
    ESSModel,
    ESSScoreOutput,
    build_score_dict,
    rank_candidate_indices,
    select_top_k_candidates,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------


@dataclass
class QueryResult:
    selected_indices: List[int]
    strategy_name: str
    metadata: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------



def _make_unlabeled_subset(dataset: Dataset, pool_state: PoolState) -> SubsetWithIndex:
    return SubsetWithIndex(dataset, pool_state.unlabeled_indices)



def _make_unlabeled_loader(
    dataset: Dataset,
    pool_state: PoolState,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    subset = _make_unlabeled_subset(dataset, pool_state)
    return build_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------



def random_query(
    pool_state: PoolState,
    budget: int,
    seed: int,
) -> QueryResult:
    """Randomly select from the current unlabeled pool."""
    if budget <= 0 or len(pool_state.unlabeled_indices) == 0:
        return QueryResult(selected_indices=[], strategy_name="random", metadata={})

    rng = random.Random(seed)
    budget = min(budget, len(pool_state.unlabeled_indices))
    selected = rng.sample(pool_state.unlabeled_indices, budget)

    return QueryResult(
        selected_indices=selected,
        strategy_name="random",
        metadata={
            "num_candidates": len(pool_state.unlabeled_indices),
            "num_selected": len(selected),
        },
    )


@torch.no_grad()
def entropy_query(
    ess_model: ESSModel,
    train_dataset: Dataset,
    pool_state: PoolState,
    budget: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool = True,
) -> QueryResult:
    """Entropy baseline using ESS logits on the full unlabeled pool.

    Since ESS exposes entropy computation naturally, we reuse it here even though
    this baseline does not use EKUS filtering or the energy term.
    """
    if budget <= 0 or len(pool_state.unlabeled_indices) == 0:
        return QueryResult(selected_indices=[], strategy_name="entropy", metadata={})

    score_output = rank_candidate_indices(
        model=ess_model,
        dataset=train_dataset,
        candidate_indices=pool_state.unlabeled_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        score_weight=0.0,  # entropy only
        pin_memory=pin_memory,
    )
    selected = select_top_k_candidates(score_output, budget)

    return QueryResult(
        selected_indices=selected,
        strategy_name="entropy",
        metadata={
            "num_candidates": len(pool_state.unlabeled_indices),
            "num_selected": len(selected),
            "scores": build_score_dict(score_output),
        },
    )


# -----------------------------------------------------------------------------
# EB-OSAL
# -----------------------------------------------------------------------------


@torch.no_grad()
def eb_osal_query(
    ekus_model: EKUSModel,
    ess_model: ESSModel,
    train_dataset: Dataset,
    pool_state: PoolState,
    budget: int,
    ekus_threshold: float,
    ess_score_weight: float,
    query_batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool = True,
) -> QueryResult:
    """Run the full EB-OSAL selection pipeline.

    Pipeline:
    1. EKUS scores the full unlabeled pool.
    2. EKUS filters likely-known candidates by energy threshold.
    3. ESS ranks those candidates using entropy + beta * energy.
    4. Top-b samples are returned.
    """
    if budget <= 0 or len(pool_state.unlabeled_indices) == 0:
        return QueryResult(selected_indices=[], strategy_name="eb_osal", metadata={})

    unlabeled_loader = _make_unlabeled_loader(
        dataset=train_dataset,
        pool_state=pool_state,
        batch_size=query_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    likely_known_indices, likely_unknown_indices, ekus_energy_by_index = filter_likely_known_samples(
        model=ekus_model,
        unlabeled_loader=unlabeled_loader,
        device=device,
        threshold=ekus_threshold,
    )

    if len(likely_known_indices) == 0:
        return QueryResult(
            selected_indices=[],
            strategy_name="eb_osal",
            metadata={
                "num_unlabeled": len(pool_state.unlabeled_indices),
                "num_likely_known": 0,
                "num_likely_unknown": len(likely_unknown_indices),
                "ekus_threshold": ekus_threshold,
                "ekus_energy": ekus_energy_by_index,
            },
        )

    score_output = rank_candidate_indices(
        model=ess_model,
        dataset=train_dataset,
        candidate_indices=likely_known_indices,
        batch_size=query_batch_size,
        num_workers=num_workers,
        device=device,
        score_weight=ess_score_weight,
        pin_memory=pin_memory,
    )
    selected = select_top_k_candidates(score_output, budget=min(budget, len(likely_known_indices)))

    return QueryResult(
        selected_indices=selected,
        strategy_name="eb_osal",
        metadata={
            "num_unlabeled": len(pool_state.unlabeled_indices),
            "num_likely_known": len(likely_known_indices),
            "num_likely_unknown": len(likely_unknown_indices),
            "num_selected": len(selected),
            "ekus_threshold": ekus_threshold,
            "ekus_energy": ekus_energy_by_index,
            "ess_scores": build_score_dict(score_output),
        },
    )


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------



def run_query_strategy(
    strategy_name: str,
    pool_state: PoolState,
    train_dataset: Dataset,
    budget: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool = True,
    ekus_model: Optional[EKUSModel] = None,
    ess_model: Optional[ESSModel] = None,
    ekus_threshold: Optional[float] = None,
    ess_score_weight: float = 0.1,
) -> QueryResult:
    """Dispatch query selection by strategy name."""
    strategy_name = strategy_name.lower()

    if strategy_name == "random":
        return random_query(pool_state=pool_state, budget=budget, seed=seed)

    if strategy_name == "entropy":
        if ess_model is None:
            raise ValueError("entropy query requires `ess_model`.")
        return entropy_query(
            ess_model=ess_model,
            train_dataset=train_dataset,
            pool_state=pool_state,
            budget=budget,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            pin_memory=pin_memory,
        )

    if strategy_name == "eb_osal":
        if ekus_model is None or ess_model is None:
            raise ValueError("eb_osal query requires both `ekus_model` and `ess_model`.")
        if ekus_threshold is None:
            raise ValueError("eb_osal query requires `ekus_threshold`.")
        return eb_osal_query(
            ekus_model=ekus_model,
            ess_model=ess_model,
            train_dataset=train_dataset,
            pool_state=pool_state,
            budget=budget,
            ekus_threshold=ekus_threshold,
            ess_score_weight=ess_score_weight,
            query_batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unsupported query strategy: {strategy_name}")