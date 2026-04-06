# Energy-Based Open-Set Active Learning

A compact PyTorch codebase for **Energy-Based Open-Set Active Learning (EB-OSAL)** for object classification.

This repository implements a dual-stage open-set active learning framework:

* **EKUS**: Energy-based Known/Unknown Separator
* **ESS**: Energy-based Sample Scorer

The code supports both:

* **2D image classification**: CIFAR-10, CIFAR-100, TinyImageNet
* **3D object classification**: ModelNet40 with PointNet

---

## Overview

Traditional active learning assumes a closed-set setting, where all unlabeled samples belong to classes of interest. In open-set active learning, the unlabeled pool contains both:

* samples from **known classes** that should be queried for annotation
* samples from **unknown classes** that should ideally be filtered out

This repository follows a dual-stage pipeline:

1. **EKUS** scores the unlabeled pool with energy and filters out samples that are likely unknown.
2. **ESS** ranks the retained likely-known samples using predictive entropy and energy.
3. The top-ranked samples are queried and added to the labeled set.
4. The process repeats over multiple active learning cycles.

---

## Repository structure

```text
energy-based-open-set-active-learning/
├── backbones/
│   ├── pointnet.py
│   └── resnet.py
├── configs/
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── default.yaml
│   ├── modelnet40.yaml
│   └── tinyimagenet.yaml
├── methods/
│   ├── active_loop.py
│   ├── ekus.py
│   ├── ess.py
│   └── query_strategies.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── data.py
├── main.py
├── requirements.txt
└── utils.py
```

---

## Installation

Create a Python environment and install the dependencies.

```bash
pip install -r requirements.txt
```

The current requirements file contains:

```text
torch>=2.0
torchvision>=0.15
numpy>=1.24
pillow>=9.0
pyyaml>=6.0
h5py>=3.8
```

---

## Datasets

### CIFAR-10 / CIFAR-100

These datasets can be downloaded automatically through `torchvision` if `download: true` is set in the config.

### TinyImageNet

Expected layout:

```text
<root>/
├── train/
├── val/
└── wnids.txt
```

Update `configs/tinyimagenet.yaml` so that `dataset.root` points to your TinyImageNet directory.

### ModelNet40

Expected layout is the common HDF5 release format, for example:

```text
<root>/
├── ply_data_train0.h5
├── ply_data_train1.h5
├── ...
├── ply_data_test0.h5
└── ...
```

If you use an `.npz` fallback format, `train.npz` and `test.npz` should contain:

* `data`: shape `[N, P, 3]`
* `label`: shape `[N]` or `[N, 1]`

Update `configs/modelnet40.yaml` so that `dataset.root` points to your ModelNet40 directory.

---

## Running the code

The main entry point is:

```bash
python main.py --config <dataset-config>
```

### Examples

Run CIFAR-10:

```bash
python main.py --config configs/cifar10.yaml
```

Run CIFAR-100:

```bash
python main.py --config configs/cifar100.yaml
```

Run TinyImageNet:

```bash
python main.py --config configs/tinyimagenet.yaml
```

Run ModelNet40:

```bash
python main.py --config configs/modelnet40.yaml
```

### Optional overrides

Use a specific device:

```bash
python main.py --config configs/cifar10.yaml --device cuda:0
```

Override the seed:

```bash
python main.py --config configs/cifar10.yaml --seed 123
```

Specify an explicit output directory:

```bash
python main.py --config configs/cifar10.yaml --output-dir outputs/exp1
```

Enable deterministic mode:

```bash
python main.py --config configs/cifar10.yaml --deterministic
```

---

## Configuration

The code uses:

* `configs/default.yaml` for shared defaults
* one dataset-specific config file for each experiment

Important config sections include:

* `dataset`
* `model`
* `active_learning`
* `training`
* `optimizer`
* `scheduler`
* `batch_size`
* `method.ekus`
* `method.ess`

### Example

```yaml
seed: 42

dataset:
  name: cifar10
  root: ./data
  download: true
  mismatch_ratio: 0.2
  initial_label_ratio: 0.01

model:
  backbone: resnet18

active_learning:
  num_cycles: 10
  query_budget: 1500
  query_strategy: eb_osal
```

---

## Outputs

Each run creates an experiment directory under `outputs/`.

Typical contents:

* `checkpoints/`: saved EKUS and ESS weights per cycle
* `logs/`: per-cycle JSON logs and run log
* `results/`: summaries and exported metrics

The main output files usually include:

* `resolved_config.yaml`
* `results/data_summary.json`
* `results/final_summary.json`
* per-cycle logs under `logs/`

---

## Notes on label mapping

This implementation uses **known-class label remapping** during training and known-only evaluation.

For example, if the known classes in the original dataset are:

```text
[2, 5, 9]
```

then the classifier is trained on remapped labels:

```text
2 -> 0
5 -> 1
9 -> 2
```

This is necessary because EKUS and ESS classifier heads are sized to the number of known classes, not the full dataset class count.

---

## Notes on the current implementation

This repository is intentionally compact. It is organized around the method itself rather than a large general-purpose framework.

Responsibilities are separated as follows:

* `data.py`: datasets, open-set splits, pool state, dataloaders
* `backbones/`: ResNet and PointNet backbones
* `methods/ekus.py`: EKUS model, losses, and training utilities
* `methods/ess.py`: ESS model, losses, scoring, and training utilities
* `methods/query_strategies.py`: query policies
* `methods/active_loop.py`: cycle-level orchestration
* `main.py`: experiment bootstrap

---

## Current limitations

A few points to keep in mind:

* this is a research codebase, not a production training framework
* dataset paths must be configured manually for TinyImageNet and ModelNet40
* runtime behavior still depends on the exact data format you provide for ModelNet40
* if you change interfaces between modules, update `main.py`, `data.py`, and `methods/active_loop.py` consistently

---

## Citation

If you use this repository in your work, please cite the corresponding paper once the final bibliographic entry is available.

```bibtex
@article{lyu2026ebosal,
  title={Energy-Based Open-Set Active Learning for Object Classification},
  author={Lyu, Zongyao and Beksi, William J.},
  journal={TBD},
  year={2026}
}
```
