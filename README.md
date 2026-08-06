# segsRoad

`segsRoad` is a shared MMSegmentation-based codebase for remote sensing
segmentation research. It is maintained as a common framework for several
related papers, while each paper keeps its method-specific description,
configuration list, and reproduction notes in a separate README.

## Project Index

| Project | Status | Task | Documentation |
|---|---|---|---|
| GeoFAD-SAM | Preparing for submission | SAM-based remote sensing semantic segmentation | [resources/docs/GeoFAD-SAM.md](resources/docs/GeoFAD-SAM.md) |
| FDMamba | Accepted | Topology-aware road extraction | [resources/docs/FDMamba.md](resources/docs/FDMamba.md) |
| SegRoadv3 | Under review | Fine-grained road extraction | [resources/docs/Segroadv3.md](resources/docs/Segroadv3.md) |

## Framework

This repository follows the MMSegmentation/MMEngine workflow:

```text
configs/                 Experiment configs
mmseg/models/            Backbones, decode heads, segmentors, and utilities
mmseg/datasets/          Dataset definitions and transforms
mmseg/evaluation/        Metrics
tools/train.py           Training entry point
tools/test.py            Evaluation entry point
demo/                    Optional inference and visualization scripts
```

Project-specific configs are grouped by method or experiment family, for
example:

```text
configs/fdmamba/
configs/segroad/
configs/lovedaSam/
configs/Samvaihingen/
configs/Sampostdam/
configs/Samchn6/
```

## Installation

Create and activate an environment:

```bash
conda create -n segsroad python=3.10 -y
conda activate segsroad
```

Install PyTorch for your CUDA version, then install the MMSegmentation
dependencies:

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
pip install -r requirements.txt
pip install -v -e .
```

Some Mamba-based configs require custom CUDA operators. Follow the
paper-specific README when a config uses Mamba, deformable scan, or selective
scan modules.

## Data

Dataset paths are defined in `configs/_base_/datasets/`. Before training or
testing, check the `data_root`, image directory, annotation directory, class
names, and label settings in the dataset config used by your experiment.

Typical MMSegmentation-style layout:

```text
data/
├── dataset_name/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── train/
│       └── val/
```

Some datasets in this repository use `img_dir/ann_dir`; others use
`images/annotations`. The config file is the source of truth.

## Training

Single-GPU training:

```bash
python tools/train.py /path/to/config.py --amp
```

Multi-GPU training:

```bash
bash tools/dist_train.sh /path/to/config.py 2 --amp
```

Replace `2` with the number of GPUs.

## Evaluation

Evaluate a trained checkpoint:

```bash
python tools/test.py /path/to/config.py /path/to/checkpoint.pth
```

Save prediction maps when needed:

```bash
python tools/test.py /path/to/config.py /path/to/checkpoint.pth --show-dir /path/to/output_dir
```

## Inference

Run inference on one image or an image directory:

```bash
python demo/image_demo_with_inferencer.py /path/to/image_or_dir /path/to/config.py --checkpoint /path/to/checkpoint.pth --output-dir /path/to/output_dir
```

## Checkpoints and External Code

Large files are intentionally excluded from Git:

```text
work_dirs/
weights/
*.pth
mmseg/models/backbones/sam3/
```

SAM-based experiments require the local SAM3 source tree and pretrained weights
to be prepared separately. See
[resources/docs/GeoFAD-SAM.md](resources/docs/GeoFAD-SAM.md) for the current
SAM-based experiment entry points.

## Repository Hygiene

This repository is used for active paper development. To keep commits readable:

- use a dedicated branch for each paper or submission stage;
- commit core method code separately from notebooks, visualizations, and
  exploratory scripts;
- do not commit checkpoints, datasets, `work_dirs/`, notebook outputs, or
  machine-specific absolute paths;
- use `git add -f <path>` only when an ignored file is intentionally part of a
  release package.

## Acknowledgements

This codebase builds on
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation), MMEngine, and
related open-source segmentation/backbone implementations. Please cite the
corresponding original projects when using external models, datasets, or
pretrained weights.
