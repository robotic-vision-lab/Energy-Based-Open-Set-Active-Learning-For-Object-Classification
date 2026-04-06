"""Main entry point for EB-OSAL.

This script is intentionally thin. It owns only high-level experiment bootstrap:
- parse arguments
- load config files
- prepare logging / output directories
- build datasets and models
- launch the active learning loop

It does not own method-specific training logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from backbones.pointnet import PointNetBackbone
from backbones.resnet import ResNetBackbone, resnet18, resnet34, resnet50
from data import build_datasets, summarize_data_state
from methods.active_loop import run_active_learning_loop
from methods.ekus import build_ekus
from methods.ess import build_ess
from utils import (
    build_experiment_name,
    count_parameters,
    get_device,
    load_config,
    log_config,
    prepare_output_dirs,
    save_json,
    save_yaml,
    set_seed,
    setup_logger,
    to_serializable,
)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EB-OSAL: Energy-Based Open-Set Active Learning")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset / experiment config YAML, e.g. configs/cifar10.yaml",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default.yaml",
        help="Path to default config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config, e.g. cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. If omitted, a timestamped experiment directory is created.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic cuDNN behavior for reproducibility.",
    )

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Config post-processing
# -----------------------------------------------------------------------------
def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    return config

# -----------------------------------------------------------------------------
# Backbone builders
# -----------------------------------------------------------------------------
def build_backbone(config: Dict[str, Any]) -> Tuple[torch.nn.Module, int]:
    """Build the configured backbone and return (module, feature_dim)."""
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    backbone_name = str(model_cfg["backbone"]).lower()
    num_classes = _infer_num_classes(dataset_cfg["name"])

    if backbone_name == "resnet18":
        net = resnet18(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        backbone = ResNetBackbone(net)
        return backbone, backbone.feature_dim

    if backbone_name == "resnet34":
        net = resnet34(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        backbone = ResNetBackbone(net)
        return backbone, backbone.feature_dim

    if backbone_name == "resnet50":
        net = resnet50(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        backbone = ResNetBackbone(net)
        return backbone, backbone.feature_dim

    if backbone_name == "pointnet":
        backbone = PointNetBackbone(
            feature_transform=bool(model_cfg.get("feature_transform", True))
        )
        return backbone, backbone.feature_dim

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def _infer_num_classes(dataset_name: str) -> int:
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "tinyimagenet":
        return 200
    if dataset_name == "modelnet40":
        return 40
    raise ValueError(f"Unsupported dataset for class-count inference: {dataset_name}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 1. Load and finalize config.
    config = load_config(args.default_config, args.config)
    config = apply_cli_overrides(config, args)

    # 2. Reproducibility and device setup.
    seed = int(config["seed"])
    set_seed(seed, deterministic=args.deterministic)
    device = get_device(config.get("device", "auto"))

    # 3. Prepare experiment directory.
    output_root = Path(config.get("output_dir", "outputs"))
    if args.output_dir is None:
        exp_name = build_experiment_name(config)
        experiment_dir = output_root / exp_name
    else:
        experiment_dir = output_root

    output_dirs = prepare_output_dirs(experiment_dir)
    logger = setup_logger(
        name="eb_osal",
        log_file=output_dirs["logs"] / "run.log",
    )

    logger.info("Starting EB-OSAL experiment")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {output_dirs['root']}")
    log_config(logger, config)

    # Save a copy of the resolved config for reproducibility.
    save_yaml(output_dirs["root"] / "resolved_config.yaml", config)

    # 4. Build data.
    data_bundle = build_datasets(config)
    data_summary = summarize_data_state(data_bundle)
    logger.info(f"Data summary: {data_summary}")
    save_json(output_dirs["results"] / "data_summary.json", to_serializable(data_summary))

    # 5. Build backbone and stage-specific models.
    # backbone, feature_dim = build_backbone(config)
    # ekus_model = build_ekus(
    #     backbone=backbone,
    #     feature_dim=feature_dim,
    #     num_classes=len(data_bundle.split.known_classes),
    # )
    # ess_model = build_ess(
    #     backbone=build_backbone(config)[0],
    #     feature_dim=feature_dim,
    #     num_classes=len(data_bundle.split.known_classes),
    # )
    ekus_backbone, ekus_feature_dim = build_backbone(config)
    ess_backbone, ess_feature_dim = build_backbone(config)

    ekus_model = build_ekus(
        backbone=ekus_backbone,
        feature_dim=ekus_feature_dim,
        num_classes=len(data_bundle.split.known_classes),
    )

    ess_model = build_ess(
        backbone=ess_backbone,
        feature_dim=ess_feature_dim,
        num_classes=len(data_bundle.split.known_classes),
    )

    logger.info(
        f"EKUS parameters: {count_parameters(ekus_model):,d} | "
        f"ESS parameters: {count_parameters(ess_model):,d}"
    )

    # 6. Launch active learning.
    result = run_active_learning_loop(
        ekus_model=ekus_model,
        ess_model=ess_model,
        data_bundle=data_bundle,
        config=config,
        device=device,
        outputs_dir=output_dirs["root"],
    )

    # 7. Save a compact final summary.
    final_summary = {
        "num_cycles": len(result.history),
        "final_num_labeled": len(result.final_pool_state.labeled_indices),
        "final_num_unlabeled": len(result.final_pool_state.unlabeled_indices),
        "history": [to_serializable(record) for record in result.history],
    }
    save_json(output_dirs["results"] / "final_summary.json", final_summary)

    if result.history:
        last = result.history[-1]
        logger.info(
            "Finished. "
            f"Last-cycle ESS accuracy: {last.ess_eval.get('accuracy', 0.0):.4f}, "
            f"labeled: {last.num_labeled}, unlabeled: {last.num_unlabeled}"
        )
    else:
        logger.info("Finished with no active learning cycles executed.")


if __name__ == "__main__":
    main()