"""Feature artifact visualizer for SAM/DINO style segmentation models.

The module is intentionally config driven:

- use any MMSegmentation config + checkpoint pair;
- collect arbitrary module outputs by name, with ``backbone`` as the default;
- trace common register-token implementations when they expose enough state;
- render norm, PCA, long-range feature repetition, position asymmetry, register
  affinity, and segmentation overlays.

It is designed to be imported from ``demo/feature_artifact_visualizer.ipynb``,
but can also be used as a small CLI for a single model/image pair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_MODEL_SPECS = [
    {
        "name": "loveda_dinov3_upernet",
        "config": "configs/dino/loveda_dino_upernet_40k.py",
        "checkpoint": [
            "work_dirs/loveda_deep_dino_upernet_40k/best_mIoU_iter_28000.pth",
            # "work_dirs/loveda_deep_dino_upernet_40k/iter_40000.pth",
            # "work_dirs/loveda_dino_upernet_40k/iter_40000.pth",
        ],
        "hook_modules": ["backbone"],
    },
    {
        "name": "loveda_sam3_upernet",
        "config": "configs/sam/loveda_sam_withoutdinoneck_upernet_40k.py",
        "checkpoint": [
            "work_dirs/loveda_sam_withoutdinoneck_upernet_40k/best_mIoU_iter_20000.pth",
            # "work_dirs/loveda_sam_withoutdinoneck_upernet_40k/iter_40000.pth",
        ],
        "hook_modules": ["backbone"],
    },
    {
        "name": "loveda_sam3_register_upernet",
        "config": "configs/sam/loveda_sam_register_upernet_40k.py",
        "checkpoint": [
            "work_dirs/loveda_sam_register_upernet_40k/best_mIoU_iter_12000.pth",
            # "work_dirs/loveda_sam_register_upernet_40k/iter_40000.pth",
        ],
        "hook_modules": ["backbone"],
    },
]


@dataclass
class ArtifactAnalysisConfig:
    """Options controlling feature artifact measurements and rendering."""

    layer_indices: Sequence[int] | str = (-1,)
    topk: int = 16
    duplicate_threshold: float = 0.92
    local_exclusion_radius: int = 1
    max_similarity_tokens: int = 4096
    similarity_chunk_size: int = 512
    high_norm_percentile: float = 99.0
    panel_dpi: int = 160
    overlay_alpha: float = 0.55
    anchor_count: int = 3
    anchor_neighbor_count: int = 8
    save_npz: bool = True


def resolve_path(path: str | Path | None, root: Path = REPO_ROOT) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    return path


def resolve_first_existing(
    value: str | Path | Sequence[str | Path] | None,
    root: Path = REPO_ROOT,
) -> Optional[Path]:
    if value is None:
        return None
    candidates: Iterable[str | Path]
    if isinstance(value, (str, Path)):
        candidates = [value]
    else:
        candidates = value
    for candidate in candidates:
        path = resolve_path(candidate, root)
        if path is not None and path.exists():
            return path
    return resolve_path(next(iter(candidates)), root) if candidates else None


def read_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def detach_nested(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: detach_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_nested(item) for item in value)
    if isinstance(value, list):
        return [detach_nested(item) for item in value]
    return value


def flatten_tensors(value: Any) -> List[torch.Tensor]:
    tensors = []
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    return tensors


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    if name in ("", "."):
        return model
    modules = dict(model.named_modules())
    if name not in modules:
        sample = ", ".join(list(modules.keys())[:30])
        raise KeyError(f"Module {name!r} was not found. First modules: {sample}")
    return modules[name]


class ModuleOutputCollector:
    """Collect outputs from named modules during a forward pass."""

    def __init__(self, model: torch.nn.Module, module_names: Sequence[str]):
        self.model = model
        self.module_names = list(module_names)
        self.outputs: Dict[str, Any] = {}
        self.handles: List[Any] = []
        self.missing: List[str] = []

    def __enter__(self) -> "ModuleOutputCollector":
        for name in self.module_names:
            try:
                module = get_module_by_name(self.model, name)
            except KeyError:
                self.missing.append(name)
                continue

            def hook(_module, _inputs, output, hook_name=name):
                self.outputs[hook_name] = detach_nested(output)

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()


class RegisterTracer:
    """Trace register tokens for common SAM3Register/DINO3Register variants.

    The tracer is best-effort and non-invasive: unsupported models simply
    produce an empty record list.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.records: List[Dict[str, Any]] = []
        self._patched: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "RegisterTracer":
        for module_name, module in self.model.named_modules():
            if hasattr(module, "_forward_global_block_with_registers"):
                self._patch_sam_register(module_name, module)
            if hasattr(module, "_forward_block") and hasattr(module, "num_register_tokens"):
                self._patch_dino_register(module_name, module)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for module, attr_name, original in reversed(self._patched):
            setattr(module, attr_name, original)

    def _record(self, module_name: str, layer_kind: str, registers: torch.Tensor) -> None:
        self.records.append(
            {
                "module": module_name,
                "layer_index": len(self.records),
                "layer_kind": layer_kind,
                "registers": registers.detach().cpu(),
            }
        )

    def _patch_sam_register(self, module_name: str, module: Any) -> None:
        original = module._forward_global_block_with_registers

        def wrapped(block, x, registers):
            out_x, out_registers = original(block, x, registers)
            self._record(module_name, "sam_global_block", out_registers)
            return out_x, out_registers

        self._patched.append((module, "_forward_global_block_with_registers", original))
        setattr(module, "_forward_global_block_with_registers", wrapped)

    def _patch_dino_register(self, module_name: str, module: Any) -> None:
        original = module._forward_block

        def wrapped(_self, block, x, rope):
            out = original(block, x, rope)
            start = int(getattr(module, "num_original_prefix_tokens", 0))
            end = int(getattr(module, "num_total_prefix_tokens", start))
            if torch.is_tensor(out) and out.ndim == 3 and end > start:
                self._record(module_name, "dino_block", out[:, start:end])
            return out

        self._patched.append((module, "_forward_block", original))
        setattr(module, "_forward_block", types.MethodType(wrapped, module))


def init_mmseg_model(config: Path, checkpoint: Optional[Path], device: str):
    from mmengine.model import revert_sync_batchnorm
    from mmseg.apis import init_model

    model = init_model(str(config), str(checkpoint) if checkpoint else None, device=device)
    if device == "cpu":
        model = revert_sync_batchnorm(model)
    return model


def infer_mmseg(model: torch.nn.Module, image_path: Path):
    from mmseg.apis import inference_model

    return inference_model(model, str(image_path))


def tensor_to_chw(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Convert a common feature tensor to CxHxW.

    Supported layouts:
    - BxCxHxW
    - BxHxWxC
    - CxHxW
    - HxWxC
    - BxNxC or NxC when N is square.
    """

    x = tensor.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
        if x.shape[0] > 8 and x.shape[0] >= x.shape[-1]:
            return x.contiguous()
        if x.shape[-1] > 8:
            return x.permute(2, 0, 1).contiguous()
        return None

    if x.ndim == 3:
        if x.shape[0] > 8 and x.shape[1] > 1 and x.shape[2] > 1:
            return x.contiguous()
        if x.shape[-1] > 8 and x.shape[0] > 1 and x.shape[1] > 1:
            return x.permute(2, 0, 1).contiguous()
        if x.shape[0] == 1 and x.shape[-1] > 8:
            x = x[0]

    if x.ndim == 2 and x.shape[-1] > 8:
        n_tokens, channels = x.shape
        side = int(math.sqrt(n_tokens))
        if side * side != n_tokens:
            return None
        return x.reshape(side, side, channels).permute(2, 0, 1).contiguous()

    return None


def feature_maps_from_outputs(outputs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for source_name, output in outputs.items():
        for idx, tensor in enumerate(flatten_tensors(output)):
            chw = tensor_to_chw(tensor)
            if chw is None:
                continue
            features.append(
                {
                    "name": f"{source_name}.{idx}",
                    "source": source_name,
                    "index": idx,
                    "tensor": chw,
                }
            )
    return features


def select_features(
    features: Sequence[Dict[str, Any]],
    layer_indices: Sequence[int] | str,
) -> List[Dict[str, Any]]:
    if layer_indices == "all":
        return list(features)
    selected = []
    for index in layer_indices:
        actual = index if index >= 0 else len(features) + index
        if 0 <= actual < len(features):
            selected.append(features[actual])
    return selected


def normalize01(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lo = np.percentile(values[finite], low)
    hi = np.percentile(values[finite], high)
    if hi <= lo:
        hi = values[finite].max()
        lo = values[finite].min()
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def resize_map(values: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    image = Image.fromarray((normalize01(values) * 255).astype(np.uint8))
    image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


def heatmap_rgb(values: np.ndarray, cmap: str = "magma") -> np.ndarray:
    normalized = normalize01(values)
    rgb = plt.get_cmap(cmap)(normalized)[..., :3]
    return (rgb * 255).astype(np.uint8)


def heatmap_only(
    heatmap: np.ndarray,
    image_hw: Tuple[int, int],
    cmap: str = "magma",
) -> np.ndarray:
    return heatmap_rgb(resize_map(heatmap, image_hw), cmap=cmap)


def blend_rgb(base_rgb: np.ndarray, color_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(
        base_rgb.astype(np.float32) * (1.0 - alpha)
        + color_rgb.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    cmap: str = "magma",
    alpha: float = 0.55,
) -> np.ndarray:
    colored = heatmap_only(heatmap, image_rgb.shape[:2], cmap=cmap)
    return blend_rgb(image_rgb, colored, alpha)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def pca_rgb(feature_chw: torch.Tensor) -> np.ndarray:
    channels, height, width = feature_chw.shape
    flat = feature_chw.permute(1, 2, 0).reshape(-1, channels).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    try:
        _, _, vectors = torch.pca_lowrank(flat, q=3, center=False, niter=4)
        projected = flat @ vectors[:, :3]
    except RuntimeError:
        _u, _s, vh = torch.linalg.svd(flat, full_matrices=False)
        projected = flat @ vh[:3].T
    projected = projected.reshape(height, width, 3).numpy()
    channels_rgb = []
    for channel in range(3):
        channels_rgb.append(normalize01(projected[..., channel], 1.0, 99.0))
    rgb = np.stack(channels_rgb, axis=-1)
    return (rgb * 255).astype(np.uint8)


def maybe_downsample_feature(
    feature_chw: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    channels, height, width = feature_chw.shape
    if height * width <= max_tokens:
        return feature_chw, (height, width)
    scale = math.sqrt(max_tokens / float(height * width))
    new_h = max(4, int(height * scale))
    new_w = max(4, int(width * scale))
    resized = F.interpolate(
        feature_chw.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized, (height, width)


def feature_similarity_stats(
    feature_chw: torch.Tensor,
    topk: int,
    duplicate_threshold: float,
    local_exclusion_radius: int,
    max_tokens: int,
    chunk_size: int,
) -> Dict[str, Any]:
    sim_feature, original_hw = maybe_downsample_feature(feature_chw, max_tokens)
    channels, height, width = sim_feature.shape
    vectors = sim_feature.permute(1, 2, 0).reshape(-1, channels).float()
    vectors = F.normalize(vectors, dim=1, eps=1e-6)
    n_tokens = vectors.shape[0]
    topk = min(topk, max(1, n_tokens - 1))

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    grid_yx = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1)
    norm_yx = torch.stack(
        [
            y.reshape(-1) / max(1.0, height - 1.0),
            x.reshape(-1) / max(1.0, width - 1.0),
        ],
        dim=1,
    )

    top_values = torch.empty((n_tokens, topk), dtype=torch.float32)
    top_indices = torch.empty((n_tokens, topk), dtype=torch.long)
    duplicate_score = torch.empty(n_tokens, dtype=torch.float32)
    asymmetry_score = torch.empty(n_tokens, dtype=torch.float32)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        sim = vectors[start:end] @ vectors.T
        row_ids = torch.arange(start, end)
        sim[torch.arange(end - start), row_ids] = -float("inf")

        if local_exclusion_radius >= 0:
            diff = (grid_yx[start:end, None, :] - grid_yx[None, :, :]).abs()
            local_mask = diff.max(dim=-1).values <= float(local_exclusion_radius)
            sim[local_mask] = -float("inf")

        values, indices = torch.topk(sim, k=topk, dim=1)
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        dists = torch.linalg.norm(
            norm_yx[start:end, None, :] - norm_yx[indices],
            dim=-1,
        )
        repeat = torch.clamp(
            (values - duplicate_threshold) / max(1e-6, 1.0 - duplicate_threshold),
            min=0.0,
            max=1.0,
        ).mean(dim=1)

        top_values[start:end] = values
        top_indices[start:end] = indices
        duplicate_score[start:end] = repeat
        asymmetry_score[start:end] = dists.mean(dim=1)

    maps = {
        "topk_similarity": top_values.mean(dim=1).reshape(height, width).numpy(),
        "position_asymmetry": asymmetry_score.reshape(height, width).numpy(),
        "feature_repetition": duplicate_score.reshape(height, width).numpy(),
        "topk_indices": top_indices,
        "downsampled_hw": (height, width),
        "original_hw": original_hw,
    }
    return maps


def register_affinity_maps(
    feature_chw: torch.Tensor,
    register_records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not register_records:
        return None
    registers = register_records[-1].get("registers")
    if registers is None or not torch.is_tensor(registers):
        return None
    registers = registers[0].float()
    channels, height, width = feature_chw.shape
    if registers.ndim != 2 or registers.shape[-1] != channels:
        return None
    patch = feature_chw.permute(1, 2, 0).reshape(-1, channels).float()
    patch = F.normalize(patch, dim=1, eps=1e-6)
    registers = F.normalize(registers, dim=1, eps=1e-6)
    sim = patch @ registers.T
    sim_maps = sim.reshape(height, width, registers.shape[0]).permute(2, 0, 1)
    return {
        "register_similarity": sim_maps.numpy(),
        "register_max_similarity": sim_maps.max(dim=0).values.numpy(),
        "record": {
            "module": register_records[-1].get("module"),
            "layer_index": register_records[-1].get("layer_index"),
            "layer_kind": register_records[-1].get("layer_kind"),
        },
    }


def palette_from_model(model: torch.nn.Module, fallback_classes: int = 20) -> np.ndarray:
    if hasattr(model, "dataset_meta"):
        palette = model.dataset_meta.get("palette")
        if palette is not None:
            return np.asarray(palette, dtype=np.uint8)
    cmap = plt.get_cmap("tab20", fallback_classes)
    return (cmap(np.arange(fallback_classes))[:, :3] * 255).astype(np.uint8)


def colorize_label_map(
    label_map: np.ndarray,
    palette: np.ndarray,
    image_hw: Optional[Tuple[int, int]] = None,
    ignore_index: int = 255,
) -> np.ndarray:
    label = np.asarray(label_map)
    if label.ndim == 3 and label.shape[-1] in (3, 4):
        color = label[..., :3].astype(np.uint8)
    else:
        label = np.squeeze(label).astype(np.int64)
        valid = label != ignore_index
        clipped = np.clip(label, 0, len(palette) - 1)
        color = palette[clipped].astype(np.uint8)
        color[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    if image_hw is not None and color.shape[:2] != image_hw:
        color = np.asarray(
            Image.fromarray(color).resize(
                (image_hw[1], image_hw[0]),
                resample=Image.Resampling.NEAREST,
            )
        )
    return color


def prediction_visuals(
    model: torch.nn.Module,
    result: Any,
    image_rgb: np.ndarray,
    alpha: float = 0.5,
) -> Optional[Dict[str, np.ndarray]]:
    if result is None or not hasattr(result, "pred_sem_seg"):
        return None
    pred = result.pred_sem_seg.data
    if torch.is_tensor(pred):
        pred = pred.squeeze().detach().cpu().numpy()
    palette = palette_from_model(
        model, fallback_classes=int(np.max(pred)) + 1 if pred.size else 1
    )
    mask = colorize_label_map(pred, palette, image_hw=image_rgb.shape[:2])
    return {"mask": mask, "overlay": blend_rgb(image_rgb, mask, alpha)}


def gt_visuals(
    gt_path: Optional[Path],
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    alpha: float = 0.5,
) -> Optional[Dict[str, np.ndarray]]:
    if gt_path is None:
        return None
    gt_path = Path(gt_path)
    if not gt_path.exists():
        warnings.warn(f"GT path does not exist: {gt_path}")
        return None
    gt = np.asarray(Image.open(gt_path))
    palette = palette_from_model(model, fallback_classes=max(int(np.max(gt)) + 1, 20))
    mask = colorize_label_map(gt, palette, image_hw=image_rgb.shape[:2])
    return {"mask": mask, "overlay": blend_rgb(image_rgb, mask, alpha)}


def save_panel(
    path: Path,
    title: str,
    items: Sequence[Tuple[str, np.ndarray]],
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(items))
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), dpi=dpi)
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (name, image) in zip(axes_array, items):
        axis.imshow(image)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(items) :]:
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_anchor_plot(
    path: Path,
    image_rgb: np.ndarray,
    stats: Mapping[str, Any],
    score_map: np.ndarray,
    anchor_count: int,
    neighbor_count: int,
    dpi: int,
) -> None:
    top_indices = stats.get("topk_indices")
    if top_indices is None:
        return
    height, width = stats["downsampled_hw"]
    flat_score = torch.from_numpy(score_map.reshape(-1).astype(np.float32))
    anchor_count = min(anchor_count, flat_score.numel())
    if anchor_count <= 0:
        return
    anchors = torch.topk(flat_score, k=anchor_count).indices.tolist()
    fig, axis = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    axis.imshow(image_rgb)
    axis.axis("off")
    colors = ["#ff2d55", "#00a8ff", "#ffb000", "#00c853"]

    for offset, anchor in enumerate(anchors):
        ay, ax = divmod(anchor, width)
        y0 = (ay + 0.5) / height * image_rgb.shape[0]
        x0 = (ax + 0.5) / width * image_rgb.shape[1]
        color = colors[offset % len(colors)]
        axis.scatter([x0], [y0], s=52, color=color, edgecolors="white", linewidths=1.2)
        neighbors = top_indices[anchor, :neighbor_count].tolist()
        for neighbor in neighbors:
            ny, nx = divmod(int(neighbor), width)
            y1 = (ny + 0.5) / height * image_rgb.shape[0]
            x1 = (nx + 0.5) / width * image_rgb.shape[1]
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=1.1, alpha=0.7)
            axis.scatter([x1], [y1], s=16, color=color, alpha=0.75)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def metric_summary(
    norm_map: np.ndarray,
    stats: Mapping[str, Any],
    registers: Optional[Mapping[str, Any]],
    high_norm_percentile: float,
) -> Dict[str, Any]:
    threshold = float(np.percentile(norm_map, high_norm_percentile))
    summary = {
        "feature_shape": list(norm_map.shape),
        "norm_mean": float(norm_map.mean()),
        "norm_std": float(norm_map.std()),
        "norm_max": float(norm_map.max()),
        "norm_high_percentile": high_norm_percentile,
        "norm_high_threshold": threshold,
        "topk_similarity_mean": float(np.mean(stats["topk_similarity"])),
        "position_asymmetry_mean": float(np.mean(stats["position_asymmetry"])),
        "feature_repetition_mean": float(np.mean(stats["feature_repetition"])),
    }
    if registers is not None:
        reg_max = registers["register_max_similarity"]
        summary.update(
            {
                "register_max_similarity_mean": float(np.mean(reg_max)),
                "register_max_similarity_max": float(np.max(reg_max)),
                "register_record": registers.get("record"),
            }
        )
    return summary


def analyze_feature_layer(
    feature: Mapping[str, Any],
    image_rgb: np.ndarray,
    output_dir: Path,
    run_name: str,
    prediction: Optional[Mapping[str, np.ndarray]],
    gt: Optional[Mapping[str, np.ndarray]],
    register_records: Sequence[Mapping[str, Any]],
    config: ArtifactAnalysisConfig,
) -> Dict[str, Any]:
    feature_name = safe_name(str(feature["name"]))
    feature_chw = feature["tensor"].float()
    norm_map = torch.linalg.norm(feature_chw, dim=0).numpy()
    pca = pca_rgb(feature_chw)
    pca_resized = np.asarray(
        Image.fromarray(pca).resize(
            (image_rgb.shape[1], image_rgb.shape[0]),
            resample=Image.Resampling.BILINEAR,
        )
    )
    stats = feature_similarity_stats(
        feature_chw=feature_chw,
        topk=config.topk,
        duplicate_threshold=config.duplicate_threshold,
        local_exclusion_radius=config.local_exclusion_radius,
        max_tokens=config.max_similarity_tokens,
        chunk_size=config.similarity_chunk_size,
    )
    registers = register_affinity_maps(feature_chw, register_records)

    layer_dir = output_dir / run_name / feature_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    norm_overlay = overlay_heatmap(
        image_rgb, norm_map, cmap="magma", alpha=config.overlay_alpha
    )
    norm_heatmap = heatmap_only(norm_map, image_rgb.shape[:2], cmap="magma")
    asym_overlay = overlay_heatmap(
        image_rgb, stats["position_asymmetry"], cmap="viridis", alpha=config.overlay_alpha
    )
    asym_heatmap = heatmap_only(
        stats["position_asymmetry"], image_rgb.shape[:2], cmap="viridis"
    )
    repetition_overlay = overlay_heatmap(
        image_rgb, stats["feature_repetition"], cmap="inferno", alpha=config.overlay_alpha
    )
    repetition_heatmap = heatmap_only(
        stats["feature_repetition"], image_rgb.shape[:2], cmap="inferno"
    )
    similarity_overlay = overlay_heatmap(
        image_rgb, stats["topk_similarity"], cmap="plasma", alpha=config.overlay_alpha
    )
    similarity_heatmap = heatmap_only(
        stats["topk_similarity"], image_rgb.shape[:2], cmap="plasma"
    )

    save_rgb(layer_dir / "input.png", image_rgb)
    save_rgb(layer_dir / "feature_norm_overlay.png", norm_overlay)
    save_rgb(layer_dir / "feature_norm_heatmap.png", norm_heatmap)
    save_rgb(layer_dir / "feature_pca_rgb.png", pca_resized)
    save_rgb(layer_dir / "position_asymmetry_overlay.png", asym_overlay)
    save_rgb(layer_dir / "position_asymmetry_heatmap.png", asym_heatmap)
    save_rgb(layer_dir / "feature_repetition_overlay.png", repetition_overlay)
    save_rgb(layer_dir / "feature_repetition_heatmap.png", repetition_heatmap)
    save_rgb(layer_dir / "topk_similarity_overlay.png", similarity_overlay)
    save_rgb(layer_dir / "topk_similarity_heatmap.png", similarity_heatmap)
    if prediction is not None:
        save_rgb(layer_dir / "prediction_mask.png", prediction["mask"])
        save_rgb(layer_dir / "prediction_overlay.png", prediction["overlay"])
    if gt is not None:
        save_rgb(layer_dir / "gt_mask.png", gt["mask"])
        save_rgb(layer_dir / "gt_overlay.png", gt["overlay"])

    panel_items = [
        ("input", image_rgb),
        (
            "prediction" if prediction is not None else "pca",
            prediction["mask"] if prediction is not None else pca_resized,
        ),
    ]
    if gt is not None:
        panel_items.append(("GT", gt["mask"]))
    panel_items.extend(
        [
            ("feature norm", norm_overlay),
            ("PCA RGB", pca_resized),
            ("position asymmetry", asym_overlay),
            ("feature repetition", repetition_overlay),
            ("top-k similarity", similarity_overlay),
        ]
    )

    register_paths = []
    register_heatmap_paths = []
    if registers is not None:
        reg_max_overlay = overlay_heatmap(
            image_rgb,
            registers["register_max_similarity"],
            cmap="cividis",
            alpha=config.overlay_alpha,
        )
        reg_max_heatmap = heatmap_only(
            registers["register_max_similarity"], image_rgb.shape[:2], cmap="cividis"
        )
        save_rgb(layer_dir / "register_max_similarity_overlay.png", reg_max_overlay)
        save_rgb(layer_dir / "register_max_similarity_heatmap.png", reg_max_heatmap)
        panel_items.append(("register max", reg_max_overlay))
        register_paths.append(str(layer_dir / "register_max_similarity_overlay.png"))
        register_heatmap_paths.append(str(layer_dir / "register_max_similarity_heatmap.png"))
        for idx, reg_map in enumerate(registers["register_similarity"]):
            reg_overlay = overlay_heatmap(
                image_rgb, reg_map, cmap="cividis", alpha=config.overlay_alpha
            )
            reg_heatmap = heatmap_only(reg_map, image_rgb.shape[:2], cmap="cividis")
            overlay_path = layer_dir / f"register_{idx:02d}_similarity_overlay.png"
            heatmap_path = layer_dir / f"register_{idx:02d}_similarity_heatmap.png"
            save_rgb(overlay_path, reg_overlay)
            save_rgb(heatmap_path, reg_heatmap)
            register_paths.append(str(overlay_path))
            register_heatmap_paths.append(str(heatmap_path))

    panel_path = layer_dir / "artifact_panel.png"
    save_panel(panel_path, f"{run_name} / {feature['name']}", panel_items, config.panel_dpi)

    combined_score = (
        normalize01(resize_map(norm_map, stats["downsampled_hw"]))
        + normalize01(stats["position_asymmetry"])
        + normalize01(stats["feature_repetition"])
    )
    save_anchor_plot(
        layer_dir / "anchor_long_range_neighbors.png",
        image_rgb,
        stats,
        combined_score,
        config.anchor_count,
        config.anchor_neighbor_count,
        config.panel_dpi,
    )
    save_anchor_plot(
        layer_dir / "anchor_long_range_neighbors_plain.png",
        np.full_like(image_rgb, 255),
        stats,
        combined_score,
        config.anchor_count,
        config.anchor_neighbor_count,
        config.panel_dpi,
    )

    metrics = metric_summary(norm_map, stats, registers, config.high_norm_percentile)
    paths = {
        "dir": str(layer_dir),
        "panel": str(panel_path),
        "input": str(layer_dir / "input.png"),
        "norm": str(layer_dir / "feature_norm_overlay.png"),
        "norm_overlay": str(layer_dir / "feature_norm_overlay.png"),
        "norm_heatmap": str(layer_dir / "feature_norm_heatmap.png"),
        "pca": str(layer_dir / "feature_pca_rgb.png"),
        "position_asymmetry": str(layer_dir / "position_asymmetry_overlay.png"),
        "position_asymmetry_overlay": str(layer_dir / "position_asymmetry_overlay.png"),
        "position_asymmetry_heatmap": str(layer_dir / "position_asymmetry_heatmap.png"),
        "feature_repetition": str(layer_dir / "feature_repetition_overlay.png"),
        "feature_repetition_overlay": str(layer_dir / "feature_repetition_overlay.png"),
        "feature_repetition_heatmap": str(layer_dir / "feature_repetition_heatmap.png"),
        "topk_similarity": str(layer_dir / "topk_similarity_overlay.png"),
        "topk_similarity_overlay": str(layer_dir / "topk_similarity_overlay.png"),
        "topk_similarity_heatmap": str(layer_dir / "topk_similarity_heatmap.png"),
        "anchor_neighbors": str(layer_dir / "anchor_long_range_neighbors.png"),
        "anchor_neighbors_plain": str(layer_dir / "anchor_long_range_neighbors_plain.png"),
        "registers": register_paths,
        "register_heatmaps": register_heatmap_paths,
    }
    if prediction is not None:
        paths["prediction_mask"] = str(layer_dir / "prediction_mask.png")
        paths["prediction"] = str(layer_dir / "prediction_overlay.png")
        paths["prediction_overlay"] = str(layer_dir / "prediction_overlay.png")
    if gt is not None:
        paths["gt_mask"] = str(layer_dir / "gt_mask.png")
        paths["gt_overlay"] = str(layer_dir / "gt_overlay.png")

    with (layer_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if config.save_npz:
        npz_data = {
            "norm": norm_map,
            "position_asymmetry": stats["position_asymmetry"],
            "feature_repetition": stats["feature_repetition"],
            "topk_similarity": stats["topk_similarity"],
        }
        if registers is not None:
            npz_data["register_similarity"] = registers["register_similarity"]
            npz_data["register_max_similarity"] = registers["register_max_similarity"]
        np.savez_compressed(layer_dir / "artifact_maps.npz", **npz_data)

    return {
        "feature": feature["name"],
        "feature_shape": list(feature_chw.shape),
        "paths": paths,
        "metrics": metrics,
    }


def analyze_mmseg_spec(
    spec: Mapping[str, Any],
    image_path: str | Path,
    gt_path: str | Path | None = None,
    output_dir: str | Path = "demo/artifact_outputs",
    device: str = "cuda:0",
    analysis_config: Optional[ArtifactAnalysisConfig] = None,
    root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    analysis_config = analysis_config or ArtifactAnalysisConfig()
    name = safe_name(str(spec.get("name", "model")))
    image_path = resolve_path(image_path, root)
    gt_path = resolve_path(gt_path, root)
    output_dir = resolve_path(output_dir, root)
    assert image_path is not None
    assert output_dir is not None

    config_path = resolve_first_existing(spec.get("config"), root)
    checkpoint_path = resolve_first_existing(spec.get("checkpoint"), root)
    missing = []
    if config_path is None or not config_path.exists():
        missing.append(f"config={spec.get('config')}")
    if spec.get("checkpoint") is not None and (checkpoint_path is None or not checkpoint_path.exists()):
        missing.append(f"checkpoint={spec.get('checkpoint')}")
    if not image_path.exists():
        missing.append(f"image={image_path}")
    if missing:
        return {
            "name": name,
            "status": "skipped",
            "reason": "missing " + ", ".join(missing),
        }

    hook_modules = spec.get("hook_modules") or ["backbone"]
    image_rgb = read_rgb(image_path)
    run_name = safe_name(f"{image_path.stem}_{name}")

    try:
        model = init_mmseg_model(config_path, checkpoint_path, device=device)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "status": "failed",
            "reason": f"model init failed: {exc}",
            "image": str(image_path),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        }

    prediction = None
    gt = None
    try:
        with torch.no_grad():
            with ModuleOutputCollector(model, hook_modules) as collector:
                with RegisterTracer(model) as register_tracer:
                    result = infer_mmseg(model, image_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "status": "failed",
            "reason": f"inference failed: {exc}",
            "image": str(image_path),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "hook_modules": list(hook_modules),
        }
    if collector.missing:
        warnings.warn(f"{name}: missing hook modules: {collector.missing}")
    features = feature_maps_from_outputs(collector.outputs)
    selected_features = select_features(features, analysis_config.layer_indices)
    try:
        prediction = prediction_visuals(model, result, image_rgb)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{name}: failed to draw prediction visuals: {exc}")
    try:
        gt = gt_visuals(gt_path, model, image_rgb)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{name}: failed to draw GT visuals: {exc}")

    if not selected_features:
        return {
            "name": name,
            "status": "skipped",
            "reason": "no feature maps collected; check hook_modules",
            "collected_hooks": list(collector.outputs.keys()),
        }

    layer_reports = []
    for feature in selected_features:
        try:
            layer_reports.append(
                analyze_feature_layer(
                    feature=feature,
                    image_rgb=image_rgb,
                    output_dir=output_dir,
                    run_name=run_name,
                    prediction=prediction,
                    gt=gt,
                    register_records=register_tracer.records,
                    config=analysis_config,
                )
            )
        except Exception as exc:  # noqa: BLE001
            layer_reports.append(
                {
                    "feature": feature["name"],
                    "feature_shape": list(feature["tensor"].shape),
                    "status": "failed",
                    "reason": f"layer analysis failed: {exc}",
                }
            )

    return {
        "name": name,
        "status": "ok",
        "image": str(image_path),
        "gt": str(gt_path) if gt_path else None,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "hook_modules": list(hook_modules),
        "register_records": [
            {
                "module": rec.get("module"),
                "layer_index": rec.get("layer_index"),
                "layer_kind": rec.get("layer_kind"),
                "shape": list(rec["registers"].shape),
            }
            for rec in register_tracer.records
        ],
        "features_collected": [
            {"name": item["name"], "shape": list(item["tensor"].shape)}
            for item in features
        ],
        "layers": layer_reports,
    }


def resolve_gt_for_image(
    image_path: str | Path,
    image_index: int,
    gt_paths: str | Path | Sequence[str | Path | None] | Mapping[str, str | Path] | None,
) -> str | Path | None:
    if gt_paths is None:
        return None
    if isinstance(gt_paths, Mapping):
        image_str = str(image_path)
        image_name = Path(image_path).name
        image_stem = Path(image_path).stem
        return (
            gt_paths.get(image_str)
            or gt_paths.get(image_name)
            or gt_paths.get(image_stem)
        )
    if isinstance(gt_paths, (str, Path)):
        return gt_paths
    if image_index < len(gt_paths):
        return gt_paths[image_index]
    return None


def run_model_grid(
    model_specs: Sequence[Mapping[str, Any]],
    image_paths: Sequence[str | Path],
    gt_paths: str | Path | Sequence[str | Path | None] | Mapping[str, str | Path] | None = None,
    output_dir: str | Path = "demo/artifact_outputs",
    device: str = "cuda:0",
    analysis_config: Optional[ArtifactAnalysisConfig] = None,
    root: Path = REPO_ROOT,
) -> List[Dict[str, Any]]:
    reports = []
    for image_index, image_path in enumerate(image_paths):
        gt_path = resolve_gt_for_image(image_path, image_index, gt_paths)
        for spec in model_specs:
            name = spec.get("name", "model")
            print(f"[artifact] analyzing {name} on {image_path}")
            report = analyze_mmseg_spec(
                spec=spec,
                image_path=image_path,
                gt_path=gt_path,
                output_dir=output_dir,
                device=device,
                analysis_config=analysis_config,
                root=root,
            )
            if report.get("status") != "ok":
                print(f"[artifact] skipped {name}: {report.get('reason')}")
            else:
                print(f"[artifact] wrote {len(report['layers'])} layer report(s)")
            reports.append(report)

    output_dir_path = resolve_path(output_dir, root)
    assert output_dir_path is not None
    output_dir_path.mkdir(parents=True, exist_ok=True)
    with (output_dir_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    return reports


def show_report_gallery(
    reports: Sequence[Mapping[str, Any]],
    key: str = "panel",
    max_items: int = 24,
) -> None:
    try:
        from IPython.display import Image as DisplayImage
        from IPython.display import display
    except ImportError:
        print("IPython is not available; open the generated PNG files directly.")
        return

    count = 0
    for report in reports:
        if report.get("status") != "ok":
            print(f"{report.get('name')}: {report.get('reason')}")
            continue
        for layer in report.get("layers", []):
            path = layer.get("paths", {}).get(key)
            if path and Path(path).exists():
                print(f"{report.get('name')} / {layer.get('feature')}: {path}")
                display(DisplayImage(filename=path))
                count += 1
                if count >= max_items:
                    return


def load_model_specs_json(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("model specs JSON must be a list of objects")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize feature artifacts.")
    parser.add_argument("--image", nargs="+", default=["demo/0.jpg"])
    parser.add_argument("--gt", nargs="*", default=None)
    parser.add_argument("--output-dir", default="demo/artifact_outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--name", default="model")
    parser.add_argument("--model-specs-json")
    parser.add_argument("--hook-module", action="append", default=None)
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--topk", type=int, default=16)
    args = parser.parse_args()

    if args.model_specs_json:
        specs = load_model_specs_json(args.model_specs_json)
    elif args.config:
        specs = [
            {
                "name": args.name,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "hook_modules": args.hook_module or ["backbone"],
            }
        ]
    else:
        specs = DEFAULT_MODEL_SPECS

    if args.all_layers:
        layer_indices: Sequence[int] | str = "all"
    else:
        layer_indices = tuple(int(item.strip()) for item in args.layers.split(",") if item.strip())

    analysis_config = ArtifactAnalysisConfig(layer_indices=layer_indices, topk=args.topk)
    reports = run_model_grid(
        specs,
        image_paths=args.image,
        gt_paths=args.gt,
        output_dir=args.output_dir,
        device=args.device,
        analysis_config=analysis_config,
    )
    ok = sum(1 for item in reports if item.get("status") == "ok")
    print(f"[artifact] complete: {ok}/{len(reports)} reports generated")


if __name__ == "__main__":
    main()
