"""Data pipeline utilities for EB-OSAL.

This module owns:
- dataset construction for CIFAR-10, CIFAR-100, TinyImageNet, and ModelNet40
- 2D / 3D transforms
- open-set known / unknown class splits
- initial labeled / unlabeled pool construction
- per-cycle pool updates
- dataloader creation for training, querying, and evaluation
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

try:
    import h5py  # Optional dependency, used for common ModelNet40 HDF5 format.
except ImportError:
    h5py = None


# -----------------------------------------------------------------------------
# Small helper dataset wrappers
# -----------------------------------------------------------------------------
class SubsetWithIndex(Dataset):
    """Subset wrapper that also returns the original dataset index.

    Returning original indices is useful during active learning because query
    strategies select from the global unlabeled pool, not just from a temporary
    minibatch-local numbering.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int]:
        original_idx = self.indices[idx]
        sample, label = self.dataset[original_idx]
        return sample, label, original_idx


# class TransformDataset(Dataset):
#     """Applies a transform on top of an existing dataset.
#
#     This wrapper is used so the same underlying split can be reused with
#     different transforms for training, evaluation, and query-time scoring.
#     """
#
#     def __init__(self, base_dataset: Dataset, transform=None) -> None:
#         self.base_dataset = base_dataset
#         self.transform = transform
#
#     def __len__(self) -> int:
#         return len(self.base_dataset)
#
#     def __getitem__(self, idx: int):
#         x, y = self.base_dataset[idx]
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, y

class LabelMappedSubsetWithIndex(Dataset):
    """Subset wrapper that remaps labels and also returns original dataset index.

    This is used when classifier outputs correspond to the known-class subset
    only. Unknown labels are mapped to `unknown_label` if encountered.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        label_map: Dict[int, int],
        unknown_label: int = -1,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.label_map = {int(k): int(v) for k, v in label_map.items()}
        self.unknown_label = int(unknown_label)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Any, int, int]:
        original_idx = self.indices[idx]
        sample, label = self.dataset[original_idx]
        mapped_label = self.label_map.get(int(label), self.unknown_label)
        return sample, mapped_label, original_idx

# -----------------------------------------------------------------------------
# TinyImageNet
# -----------------------------------------------------------------------------
class TinyImageNetDataset(Dataset):
    """Minimal TinyImageNet dataset reader.

    Expected directory layout:
        root/
            train/
                n01443537/
                    images/*.JPEG
            val/
                images/*.JPEG
                val_annotations.txt
            wnids.txt
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"TinyImageNet wnids.txt not found at: {wnids_path}")

        self.classes = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        if split == "train":
            self._build_train_samples()
        elif split in {"val", "valid", "validation"}:
            self._build_val_samples()
        else:
            raise ValueError(f"Unsupported TinyImageNet split: {split}")

        self.targets = [label for _, label in self.samples]

    def _build_train_samples(self) -> None:
        train_root = self.root / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"TinyImageNet train directory not found at: {train_root}")

        for cls_name in self.classes:
            image_dir = train_root / cls_name / "images"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.JPEG")):
                self.samples.append((image_path, self.class_to_idx[cls_name]))

    def _build_val_samples(self) -> None:
        val_root = self.root / "val"
        image_dir = val_root / "images"
        annotations = val_root / "val_annotations.txt"
        if not image_dir.exists() or not annotations.exists():
            raise FileNotFoundError("TinyImageNet validation files are missing.")

        image_to_class: Dict[str, str] = {}
        for line in annotations.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                image_to_class[parts[0]] = parts[1]

        for image_path in sorted(image_dir.glob("*.JPEG")):
            cls_name = image_to_class.get(image_path.name)
            if cls_name is None or cls_name not in self.class_to_idx:
                continue
            self.samples.append((image_path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# -----------------------------------------------------------------------------
# ModelNet40
# -----------------------------------------------------------------------------

class ModelNet40Dataset(Dataset):
    """ModelNet40 reader supporting common HDF5 format and NPZ fallback.

    Common HDF5 layout:
        root/
            ply_data_train*.h5
            ply_data_test*.h5

    NPZ fallback expects arrays named `data` and `label`.
    Data shape should be [N, P, 3].
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_points: int = 1024,
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment

        data, labels = self._load_data()
        self.data = data.astype(np.float32)
        self.targets = labels.astype(np.int64).tolist()

        unique_labels = sorted(set(self.targets))
        self.classes = [str(i) for i in unique_labels]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pattern = "ply_data_train*.h5" if self.split == "train" else "ply_data_test*.h5"
        h5_files = sorted(self.root.glob(pattern))

        if h5_files:
            if h5py is None:
                raise ImportError("h5py is required to read ModelNet40 HDF5 files.")

            all_data, all_labels = [], []
            for file_path in h5_files:
                with h5py.File(file_path, "r") as f:
                    all_data.append(f["data"][:])
                    all_labels.append(f["label"][:].reshape(-1))
            return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)

        npz_name = "train.npz" if self.split == "train" else "test.npz"
        npz_path = self.root / npz_name
        if npz_path.exists():
            arr = np.load(npz_path)
            return arr["data"], arr["label"].reshape(-1)

        raise FileNotFoundError(
            f"Could not find ModelNet40 files under {self.root}. "
            f"Expected HDF5 files matching {pattern} or {npz_name}."
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        points = np.array(self.data[idx][: self.num_points], copy=True)
        label = self.targets[idx]

        if self.normalize:
            points = normalize_point_cloud(points)
        if self.augment and self.split == "train":
            points = random_rotate_point_cloud_z(points)
            points = jitter_point_cloud(points)

        return torch.from_numpy(points).float(), int(label)

# -----------------------------------------------------------------------------
# Point-cloud transforms / processing helpers
# -----------------------------------------------------------------------------
def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points

def random_rotate_point_cloud_z(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rotation.T

def jitter_point_cloud(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip).astype(np.float32)
    return points + noise

# -----------------------------------------------------------------------------
# Open-set split and pool bookkeeping
# -----------------------------------------------------------------------------
@dataclass
class OpenSetSplit:
    known_classes: List[int]
    unknown_classes: List[int]
    known_class_to_label: Dict[int, int]
    label_to_known_class: Dict[int, int]
    train_known_indices: List[int]
    train_unknown_indices: List[int]
    test_known_indices: List[int]
    test_unknown_indices: List[int]


@dataclass
class PoolState:
    labeled_indices: List[int]
    unlabeled_indices: List[int]
    queried_history: List[List[int]]

    def num_labeled(self) -> int:
        return len(self.labeled_indices)

    def num_unlabeled(self) -> int:
        return len(self.unlabeled_indices)

def build_known_unknown_class_split(
    num_classes: int,
    mismatch_ratio: float,
    seed: int,
    known_classes: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Construct known / unknown class partitions.

    In this codebase, mismatch ratio is interpreted the same way as in the paper:
    it specifies the proportion of known classes retained in the open-set setup.
    For example, on CIFAR-10, mismatch_ratio=0.2 means 2 known classes and 8
    unknown classes.
    """
    all_classes = list(range(num_classes))

    if known_classes is not None:
        known = sorted(list(known_classes))
        unknown = sorted([c for c in all_classes if c not in known])
        return known, unknown

    num_known = max(1, int(round(num_classes * mismatch_ratio)))
    rng = random.Random(seed)
    known = sorted(rng.sample(all_classes, num_known))
    unknown = sorted([c for c in all_classes if c not in known])
    return known, unknown

def indices_for_classes(targets: Sequence[int], classes: Sequence[int]) -> List[int]:
    class_set = set(classes)
    return [idx for idx, y in enumerate(targets) if int(y) in class_set]

def build_open_set_split(
    train_targets: Sequence[int],
    test_targets: Sequence[int],
    num_classes: int,
    mismatch_ratio: float,
    seed: int,
    known_classes: Optional[Sequence[int]] = None,
) -> OpenSetSplit:
    known, unknown = build_known_unknown_class_split(
        num_classes=num_classes,
        mismatch_ratio=mismatch_ratio,
        seed=seed,
        known_classes=known_classes,
    )

    known_class_to_label = {int(cls): i for i, cls in enumerate(known)}
    label_to_known_class = {i: int(cls) for i, cls in enumerate(known)}

    return OpenSetSplit(
        known_classes=known,
        unknown_classes=unknown,
        known_class_to_label=known_class_to_label,
        label_to_known_class=label_to_known_class,
        train_known_indices=indices_for_classes(train_targets, known),
        train_unknown_indices=indices_for_classes(train_targets, unknown),
        test_known_indices=indices_for_classes(test_targets, known),
        test_unknown_indices=indices_for_classes(test_targets, unknown),
    )

def initialize_active_learning_pool(
    train_targets: Sequence[int],
    known_classes: Sequence[int],
    initial_label_ratio: float,
    seed: int,
) -> PoolState:
    """Create initial labeled and unlabeled pools from known-class training data.

    Only known-class samples participate in the labeled pool initialization.
    Unknown-class samples remain in the unlabeled pool as part of the open-set AL
    problem.
    """

    rng = random.Random(seed)
    known_set = set(known_classes)

    per_class_indices: Dict[int, List[int]] = {cls: [] for cls in known_classes}
    unlabeled_indices: List[int] = []

    for idx, y in enumerate(train_targets):
        y = int(y)
        if y in known_set:
            per_class_indices[y].append(idx)
        unlabeled_indices.append(idx)

    labeled_indices: List[int] = []
    # for cls, cls_indices in per_class_indices.items():
    #     if not cls_indices:
    #         continue
    #     n_select = max(1, int(round(len(cls_indices) * initial_label_ratio)))
    #     n_select = min(n_select, len(cls_indices))
    #     labeled_indices.extend(rng.sample(cls_indices, n_select))
    for cls_indices in per_class_indices.values():
        if not cls_indices:
            continue
        n_select = max(1, int(round(len(cls_indices) * initial_label_ratio)))
        n_select = min(n_select, len(cls_indices))
        labeled_indices.extend(rng.sample(cls_indices, n_select))

    labeled_set = set(labeled_indices)
    remaining_unlabeled = [idx for idx in unlabeled_indices if idx not in labeled_set]

    return PoolState(
        labeled_indices=sorted(labeled_indices),
        unlabeled_indices=remaining_unlabeled,
        queried_history=[],
    )

def update_pool_state(pool_state: PoolState, queried_indices: Sequence[int]) -> PoolState:
    queried_set = set(queried_indices)
    new_labeled = sorted(set(pool_state.labeled_indices).union(queried_set))
    new_unlabeled = [idx for idx in pool_state.unlabeled_indices if idx not in queried_set]
    new_history = copy.deepcopy(pool_state.queried_history)
    new_history.append(list(queried_indices))
    return PoolState(
        labeled_indices=new_labeled,
        unlabeled_indices=new_unlabeled,
        queried_history=new_history,
    )


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def build_image_transforms(dataset_name: str, train: bool = True):
    dataset_name = dataset_name.lower()

    if dataset_name in {"cifar10", "cifar100"}:
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    if dataset_name == "tinyimagenet":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4802, 0.4481, 0.3975),
                        std=(0.2302, 0.2265, 0.2262),
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4802, 0.4481, 0.3975),
                    std=(0.2302, 0.2265, 0.2262),
                ),
            ]
        )

    raise ValueError(f"Unsupported image dataset for transforms: {dataset_name}")


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------
@dataclass
class DataBundle:
    dataset_name: str
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    split: OpenSetSplit
    pool_state: PoolState


def build_datasets(config: Dict[str, Any]) -> DataBundle:
    """Build train/test datasets and open-set AL state from config.

    Expected config keys:
        dataset.name
        dataset.root
        dataset.mismatch_ratio
        dataset.initial_label_ratio
        seed

    Optional:
        dataset.known_classes
        dataset.num_points
    """

    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"].lower()
    root = dataset_cfg["root"]
    mismatch_ratio = float(dataset_cfg["mismatch_ratio"])
    initial_label_ratio = float(dataset_cfg["initial_label_ratio"])
    seed = int(config["seed"])
    known_classes = dataset_cfg.get("known_classes")

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=dataset_cfg.get("download", True),
            transform=build_image_transforms("cifar10", train=True),
        )
        test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=dataset_cfg.get("download", True),
            transform=build_image_transforms("cifar10", train=False),
        )
        num_classes = 10
        train_targets = train_dataset.targets
        test_targets = test_dataset.targets

    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=dataset_cfg.get("download", True),
            transform=build_image_transforms("cifar100", train=True),
        )
        test_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=dataset_cfg.get("download", True),
            transform=build_image_transforms("cifar100", train=False),
        )
        num_classes = 100
        train_targets = train_dataset.targets
        test_targets = test_dataset.targets

    elif dataset_name == "tinyimagenet":
        train_dataset = TinyImageNetDataset(
            root=root,
            split="train",
            transform=build_image_transforms("tinyimagenet", train=True),
        )
        test_dataset = TinyImageNetDataset(
            root=root,
            split="val",
            transform=build_image_transforms("tinyimagenet", train=False),
        )
        num_classes = 200
        train_targets = train_dataset.targets
        test_targets = test_dataset.targets

    elif dataset_name == "modelnet40":
        num_points = int(dataset_cfg.get("num_points", 1024))
        train_dataset = ModelNet40Dataset(
            root=root,
            split="train",
            num_points=num_points,
            normalize=True,
            augment=dataset_cfg.get("augment", True),
        )
        test_dataset = ModelNet40Dataset(
            root=root,
            split="test",
            num_points=num_points,
            normalize=True,
            augment=False,
        )
        num_classes = 40
        train_targets = train_dataset.targets
        test_targets = test_dataset.targets

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    split = build_open_set_split(
        train_targets=train_targets,
        test_targets=test_targets,
        num_classes=num_classes,
        mismatch_ratio=mismatch_ratio,
        seed=seed,
        known_classes=known_classes,
    )

    pool_state = initialize_active_learning_pool(
        train_targets=train_targets,
        known_classes=split.known_classes,
        initial_label_ratio=initial_label_ratio,
        seed=seed,
    )

    return DataBundle(
        dataset_name=dataset_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        split=split,
        pool_state=pool_state,
    )


# -----------------------------------------------------------------------------
# Dataloader builders
# -----------------------------------------------------------------------------
def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

# def build_labeled_loader(
#     train_dataset: Dataset,
#     pool_state: PoolState,
#     batch_size: int,
#     num_workers: int = 4,
# ) -> DataLoader:
#     subset = SubsetWithIndex(train_dataset, pool_state.labeled_indices)
#     return build_loader(
#         subset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )

def build_labeled_loader(
    train_dataset: Dataset,
    pool_state: PoolState,
    known_class_to_label: Dict[int, int],
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Remaps known labels to 0...K-1 for training when classifier is trained on known classes only."""
    subset = LabelMappedSubsetWithIndex(
        train_dataset,
        pool_state.labeled_indices,
        label_map=known_class_to_label,
    )
    return build_loader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

def build_unlabeled_loader(
    train_dataset: Dataset,
    pool_state: PoolState,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    subset = SubsetWithIndex(train_dataset, pool_state.unlabeled_indices)
    return build_loader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

# def build_test_loader(
#     test_dataset: Dataset,
#     split: OpenSetSplit,
#     batch_size: int,
#     num_workers: int = 4,
#     known_only: bool = True,
# ) -> DataLoader:
#     indices = split.test_known_indices if known_only else list(range(len(test_dataset)))
#     subset = SubsetWithIndex(test_dataset, indices)
#     return build_loader(
#         subset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )

def build_test_loader(
    test_dataset: Dataset,
    split: OpenSetSplit,
    batch_size: int,
    num_workers: int = 4,
    known_only: bool = True,
    remap_known_labels: bool = True,
) -> DataLoader:
    indices = split.test_known_indices if known_only else list(range(len(test_dataset)))
    if known_only and remap_known_labels: # Remap known labels
        subset = LabelMappedSubsetWithIndex(
            test_dataset,
            indices,
            label_map=split.known_class_to_label,
        )
    else:
        subset = SubsetWithIndex(test_dataset, indices)
    return build_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

# def build_cycle_dataloaders(
#     data_bundle: DataBundle,
#     batch_size: int,
#     num_workers: int = 4,
# ) -> Dict[str, DataLoader]:
#     return {
#         "labeled": build_labeled_loader(
#             data_bundle.train_dataset,
#             data_bundle.pool_state,
#             batch_size=batch_size,
#             num_workers=num_workers,
#         ),
#         "unlabeled": build_unlabeled_loader(
#             data_bundle.train_dataset,
#             data_bundle.pool_state,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             shuffle=False,
#         ),
#         "test_known": build_test_loader(
#             data_bundle.test_dataset,
#             data_bundle.split,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             known_only=True,
#         ),
#     }

def build_cycle_dataloaders(
    data_bundle: DataBundle,
    batch_size: int,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    return {
        "labeled": build_labeled_loader(
            data_bundle.train_dataset,
            data_bundle.pool_state,
            data_bundle.split.known_class_to_label,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        "unlabeled": build_unlabeled_loader(
            data_bundle.train_dataset,
            data_bundle.pool_state,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        "test_known": build_test_loader(
            data_bundle.test_dataset,
            data_bundle.split,
            batch_size=batch_size,
            num_workers=num_workers,
            known_only=True,
            remap_known_labels=True,
        ),
    }


# -----------------------------------------------------------------------------
# Convenience utilities for AL code
# -----------------------------------------------------------------------------
def get_targets(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return [int(x) for x in targets]
    raise AttributeError("Dataset does not expose a `targets` attribute.")


# def get_labels_for_indices(targets: Sequence[int], indices: Sequence[int]) -> List[int]:
#     return [int(targets[idx]) for idx in indices]


def split_unlabeled_by_known_unknown(
    unlabeled_indices: Sequence[int],
    targets: Sequence[int],
    known_classes: Sequence[int],
) -> Tuple[List[int], List[int]]:
    known_set = set(known_classes)
    likely_known_part = [idx for idx in unlabeled_indices if int(targets[idx]) in known_set]
    likely_unknown_part = [idx for idx in unlabeled_indices if int(targets[idx]) not in known_set]
    return likely_known_part, likely_unknown_part


def summarize_data_state(data_bundle: DataBundle) -> Dict[str, Any]:
    train_targets = get_targets(data_bundle.train_dataset)
    unlabeled_known, unlabeled_unknown = split_unlabeled_by_known_unknown(
        data_bundle.pool_state.unlabeled_indices,
        train_targets,
        data_bundle.split.known_classes,
    )
    return {
        "dataset_name": data_bundle.dataset_name,
        "num_classes": data_bundle.num_classes,
        "known_classes": data_bundle.split.known_classes,
        "unknown_classes": data_bundle.split.unknown_classes,

        "known_class_to_label": data_bundle.split.known_class_to_label,
        "label_to_known_class": data_bundle.split.label_to_known_class,

        "num_train_known": len(data_bundle.split.train_known_indices),
        "num_train_unknown": len(data_bundle.split.train_unknown_indices),
        "num_test_known": len(data_bundle.split.test_known_indices),
        "num_test_unknown": len(data_bundle.split.test_unknown_indices),
        "num_labeled": len(data_bundle.pool_state.labeled_indices),
        "num_unlabeled": len(data_bundle.pool_state.unlabeled_indices),
        "num_unlabeled_known_gt": len(unlabeled_known),
        "num_unlabeled_unknown_gt": len(unlabeled_unknown),
    }
