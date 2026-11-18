#!/usr/bin/env python3
"""Interactive viewer for the 3D DINOv2 training augmentations, on one series.

Shows what training does to a series, with the real code: the series is read by ``MIDataset`` (KneeNo's
reader, ``series_depth`` resampling, voxel spacing) and goes through ``DINOv2ViTTransform``, built from a run
config's ``transform:`` block by lightly-train's own ``get_transform_args`` / ``get_transform``. Only two things
differ from training:

* **One augmentation at a time.** The one under test runs with ``prob = 1``; every other one is off. For the
  blur and the sharpen, the local view uses the config's ``local_view`` block, as in training.
* **The crop is fixed.** ``CenterCrop3D`` is the training crop (``AnisotropyTrackingRandomResizedCrop3D``) with
  centred placement and no aspect jitter. Its ``scale`` slider is the crop's volume fraction, the quantity of the
  config's ``random_resize``. Resampling and the view-spacing tag are the training crop's own code, so the depth
  sigma of the blur and the sharpen is scaled exactly as in training.

Three columns: the whole series (with the global view's crop as a solid red box and the local view's as a dashed
one) and the two views. The top row is one slice (``D``), the bottom row one column (``W``), i.e. a cut along
depth; the yellow lines mark where the other row cuts. Images are drawn with their physical aspect ratio (voxel
spacing; uncheck "Physical aspect" for square voxels). The view titles show the view's voxel spacing, its
``scale_z = sqrt(h * w) / d`` and what the augmentation drew. "Resample" draws new random values, "Show original"
shows the views without the augmentation. The radio buttons set the resize interpolation, in-plane (H/W) and
along depth (D) separately: the config's ``resize_interpolation`` and ``resize_upscale_interpolation`` (used on
enlarged axes; "off" = ``null``, the first everywhere), and the rotation's ``interpolation`` when testing it.

Run from the repo root::

    .venv/bin/python tools/transform_viewer.py <data_root>/<case_id>/<series> gaussian_blur --spacing 0.3 0.3 3.0

``--spacing`` is the series' voxel spacing in mm, ``[H, W, D]`` like the metadata's ``data_spacing``. Without it the
series has none, and the pipeline falls back to ``1 / shape`` (see ``monai_wrappers.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from lightly_train._commands import train_helpers
from lightly_train._data.mi_dataset import MIDataset, reseed_randomizables
from lightly_train._transforms.monai_wrappers import (
    VOXEL_SPACING_META_KEY,
    AlphaRandHistogramShift,
    AnisotropyAwareRandGaussianSharpen,
    AnisotropyAwareRandGaussianSmooth,
    AnisotropyTrackingRandomResizedCrop3D,
)
from lightly_train._transforms.random_resized_crop import (
    _WEIGHT_FNS,
    CropParams3D,
    parse_interpolation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "cluster" / "configs" / "pretrain-MI-vitb14-24f.yaml"

#: The augmentations of the config's ``transform:`` block, in pipeline order.
AUGMENTATIONS = (
    "random_rotation",
    "histogram_shift",
    "adjust_contrast",
    "gaussian_blur",
    "gaussian_sharpen",
    "gibbs_noise",
    "gaussian_noise",
)
#: Augmentations whose local view has its own block (``transform.local_view.<name>``).
LOCAL_OVERRIDES = ("gaussian_blur", "gaussian_sharpen")
#: The second global view's own blocks (``transform.global_view_1``); that view isn't shown.
GLOBAL_VIEW_1 = ("gaussian_blur", "gaussian_sharpen", "gibbs_noise")
#: Starting values for an augmentation the config leaves off; the args classes' defaults fill in the rest.
FALLBACK_ARGS: dict[str, dict[str, Any]] = {"random_rotation": {"degrees": 15.0}}

#: Slider bounds (vmin, vmax) per config field. A config value outside them widens them.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "degrees": (-180.0, 180.0),
    "alpha": (0.0, 1.0),  # histogram shift and Gibbs noise
    "num_control_points": (3, 30),
    "gamma": (0.1, 4.5),
    "sigma_range": (0.0, 5.0),
    "sigma1": (0.0, 5.0),
    "sigma2": (0.0, 5.0),
    "mean": (-50.0, 50.0),  # raw [0, 255] intensity units
    "std": (0.0, 50.0),
}
#: Overrides of ``FIELD_RANGES`` per (augmentation, field).
AUG_FIELD_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("gaussian_sharpen", "alpha"): (0.0, 50.0)
}

VIEWS = ("global", "local")
_SLICE_NAME = re.compile(r"^\d{3}\.jpeg$")


# --------------------------------------------------------------------------- the pipeline


class CenterCrop3D(AnisotropyTrackingRandomResizedCrop3D):
    """The training crop, placed in the centre without aspect jitter.

    ``scale`` is the crop's volume fraction, as in the config's ``random_resize``. The sizes are what the training
    crop draws for that scale and an aspect jitter of 1 (``_sizes_from_ratios``, anchored to the volume's own
    aspect ratio). Only ``get_params`` is replaced: the resampling and the view-spacing tag are the training crop's.
    """

    def __init__(
        self, crop: AnisotropyTrackingRandomResizedCrop3D, scale: float
    ) -> None:
        super().__init__(
            size=crop.size,
            scale=(scale, scale),
            ratio=(1.0, 1.0),
            interpolation=crop.interpolation,
            upscale_interpolation=crop.upscale_interpolation,
            output_dtype=crop.output_dtype,
        )
        self.last_params: CropParams3D | None = None
        self.last_spacing: tuple[float, float, float] | None = None

    def get_params(
        self, volume_shape: Sequence[int], foreground_box: Any = None
    ) -> CropParams3D:
        height, width, depth = (int(s) for s in volume_shape[-3:])
        sizes = self._sizes_from_ratios(
            self.scale[0] * height * width * depth, height / depth, width / depth
        )
        h, w, d = (min(max(s, 1), n) for s, n in zip(sizes, (height, width, depth)))
        self.last_params = CropParams3D(
            h_start=(height - h) // 2,
            w_start=(width - w) // 2,
            d_start=(depth - d) // 2,
            height=h,
            width=w,
            depth=d,
        )
        return self.last_params

    def __call__(self, data: Any) -> Any:
        out = super().__call__(data)
        self.last_spacing = tuple(out.meta[VOXEL_SPACING_META_KEY])
        return out


def load_config(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``transform:`` and ``data:`` blocks of a run config (``{}`` each without one). Not env-expanded: neither
    block needs it."""
    if path is None:
        return {}, {}
    config = yaml.safe_load(path.read_text()) or {}
    return dict(config.get("transform") or {}), dict(config.get("data") or {})


def isolate(transform_cfg: dict[str, Any], augmentation: str | None) -> dict[str, Any]:
    """``transform_cfg`` with only ``augmentation`` on (``prob = 1``), one local view and no foreground crop.

    ``None`` turns every augmentation off (crop, resize and normalize only).
    """
    cfg = copy.deepcopy(transform_cfg)
    for name in AUGMENTATIONS:
        cfg[name] = None
    if augmentation is not None:
        cfg[augmentation] = {
            **(transform_cfg.get(augmentation) or FALLBACK_ARGS.get(augmentation, {})),
            "prob": 1.0,
        }
    cfg["global_view_1"] = {name: None for name in GLOBAL_VIEW_1}  # not shown
    local = dict(cfg.get("local_view") or {})
    local["num_views"] = 1
    for name in LOCAL_OVERRIDES:
        local[name] = (
            {**(local.get(name) or {}), "prob": 1.0} if name == augmentation else None
        )
    cfg["local_view"] = local
    cfg["crop_foreground"] = None  # the crop is centred, the tissue box would be unused
    return cfg


def resolve(cfg: dict[str, Any]) -> Any:
    """The validated ``DINOv2ViTTransformArgs``, exactly as ``pretrain`` resolves its ``transform_args``."""
    return train_helpers.get_transform_args(method="dinov2", transform_args=cfg)


def build_transform(
    cfg: dict[str, Any], scales: dict[str, float]
) -> tuple[Any, dict[str, Any]]:
    """The training transform for ``cfg`` with the global and the local view only, each with a ``CenterCrop3D``.

    Returns ``(transform, views)``: ``views[name]`` is that view's ``ViewTransform``.
    """
    args = resolve(cfg)
    transform = train_helpers.get_transform(
        method="dinov2", transform_args_resolved=args
    )
    views = {"global": transform.transforms[0], "local": transform.transforms[2]}
    for name, view in views.items():
        steps = view.transform.transforms
        view.transform.transforms = (CenterCrop3D(steps[0], scales[name]), *steps[1:])
    # DINOTransform.__call__ maps over this list: the second global view and further local views are skipped.
    transform.transforms = [views["global"], views["local"]]
    return transform, views


def load_series(
    series_dir: Path,
    spacing: Sequence[float] | None,
    series_depth: int,
    resample_mode: str,
) -> tuple[MIDataset, Path]:
    """An ``MIDataset`` over this one series, through a metadata file in a temporary directory.

    ``spacing`` is ``[H, W, D]`` in mm, written as the series' ``data_spacing``. Returns the dataset and the
    temporary directory (the caller keeps it alive).
    """
    names = sorted(p.name for p in series_dir.iterdir() if _SLICE_NAME.match(p.name))
    if not names or names != [f"{i:03d}.jpeg" for i in range(len(names))]:
        raise SystemExit(
            f"{series_dir}: expected the KneeNo layout, slices 000.jpeg, 001.jpeg, ... with no gaps"
        )
    with Image.open(series_dir / names[0]) as image:
        width, height = image.size
    entry: dict[str, Any] = {"dimensions": [height, width, len(names)]}
    if spacing is not None:
        entry["data_spacing"] = [float(s) for s in spacing]
    tmp = Path(tempfile.mkdtemp(prefix="transform_viewer_"))
    meta = tmp / "meta.json"
    meta.write_text(
        json.dumps({"train": {series_dir.parent.name: {series_dir.name: entry}}})
    )
    dataset = MIDataset(
        data_root={"internal": series_dir.parent.parent, "msk": None},
        data_meta={"internal": meta, "msk": None},
        transform=lambda input: [],  # replaced by Renderer.render
        series_depth=series_depth,
        resample_mode=resample_mode,
        min_series_len=1,
    )
    return dataset, tmp


@dataclass
class Rendered:
    """One rendering. Arrays are ``(D, H, W)`` in display units (raw intensity / 255); crops, spacings ``(H, W, D)``."""

    volume: np.ndarray
    volume_spacing: tuple[float, float, float]
    views: dict[str, np.ndarray] = field(default_factory=dict)
    crops: dict[str, CropParams3D] = field(default_factory=dict)
    spacings: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)


class Renderer:
    """Runs the training transform on the series and collects what the viewer shows."""

    def __init__(self, dataset: MIDataset, has_spacing: bool) -> None:
        self.dataset = dataset
        self.has_spacing = has_spacing
        volume = (
            dataset._core[0][0].numpy().astype(np.float32)
        )  # (D, H, W), after series_depth resampling
        self.volume = volume / 255.0
        spacing = dataset._core.voxel_spacing(0)  # (D, H, W) or None
        if spacing is None:  # the pipeline's fallback: 1 / shape
            d, h, w = volume.shape
            self.volume_spacing = (1.0 / h, 1.0 / w, 1.0 / d)
        else:
            self.volume_spacing = (spacing[1], spacing[2], spacing[0])

    def render(
        self, cfg: dict[str, Any], scales: dict[str, float], seed: int
    ) -> Rendered:
        transform, views = build_transform(cfg, scales)
        reseed_randomizables(transform, rng=np.random.RandomState(seed))
        torch.manual_seed(seed)
        self.dataset.transform = transform
        outputs = self.dataset[0][
            "views"
        ]  # MIDataset -> DINOTransform.__call__, as in training
        normalize = transform.transform_args.normalize
        mean, std = normalize.mean[0], normalize.std[0]
        rendered = Rendered(volume=self.volume, volume_spacing=self.volume_spacing)
        for (name, view), output in zip(views.items(), outputs):
            crop = view.transform.transforms[0]
            rendered.views[name] = (
                output[0].numpy() * std + mean
            )  # undo NormalizeIntensity: raw / 255
            rendered.crops[name] = crop.last_params
            rendered.spacings[name] = crop.last_spacing
            rendered.info[name] = self._describe(view, crop)
        return rendered

    def _describe(self, view: Any, crop: CenterCrop3D) -> str:
        p = crop.last_params
        h, w, d = crop.last_spacing
        unit = " mm" if self.has_spacing else " (rel.)"
        lines = [
            f"{'x'.join(map(str, crop.size))} from crop {p.height}x{p.width}x{p.depth} (scale {crop.scale[0]:.3g})",
            f"spacing {h:.3g} x {w:.3g} x {d:.3g}{unit}, scale_z {math.sqrt(h * w) / d:.3f}",
        ]
        for step in view.transform.transforms[1:]:
            text = _describe_draw(step)
            if text:
                lines.append(text)
        return "\n".join(lines)


def _describe_draw(step: Any) -> str | None:
    """What a random step drew on its last call, or ``None`` for a step without anything to show."""
    if isinstance(step, AnisotropyAwareRandGaussianSmooth):
        lo, hi = step.sigma_z
        return f"sigma x {step.x:.3g}, y {step.y:.3g}, z {step.z:.3g} (z range {lo:.3g}-{hi:.3g})"
    if isinstance(step, AnisotropyAwareRandGaussianSharpen):
        return f"sigma1 x {step.x1:.3g}, z {step.z1:.3g}; sigma2 x {step.x2:.3g}, z {step.z2:.3g}; alpha {step.a:.3g}"
    if isinstance(step, AlphaRandHistogramShift):
        return f"alpha {step.alpha:.3g}"
    name = type(step).__name__
    if name == "RandRotate":
        return f"angle {math.degrees(step.z):.3g} deg"
    if name == "RandAdjustContrast":
        return f"gamma {step.gamma_value:.3g}"
    if name == "RandGibbsNoise":
        return f"alpha {step.sampled_alpha:.3g}"
    return None


# --------------------------------------------------------------------------- parameter rows


@dataclass
class ParamRow:
    """One slider: a numeric config field (``index`` set for one end of a ``(lo, hi)`` range)."""

    path: tuple[
        str, ...
    ]  # the block, e.g. ("gaussian_blur",) or ("local_view", "gaussian_blur")
    name: str
    index: int | None
    vmin: float
    vmax: float
    is_int: bool

    @property
    def label(self) -> str:
        prefix = "local " if self.path[0] == "local_view" else ""
        suffix = {None: "", 0: " lo", 1: " hi"}[self.index]
        return f"{prefix}{self.name}{suffix}"

    def get(self, cfg: dict[str, Any]) -> float:
        value = _block(cfg, self.path)[self.name]
        return value if self.index is None else value[self.index]

    def set(self, cfg: dict[str, Any], value: float) -> None:
        block = _block(cfg, self.path)
        value = int(round(value)) if self.is_int else float(value)
        if self.index is None:
            block[self.name] = value
        else:
            pair = list(block[self.name])
            pair[self.index] = value
            block[self.name] = tuple(pair)


def _block(cfg: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    for key in path:
        cfg = cfg[key]
    return cfg


def _resolved_block(args: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        args = getattr(args, key)
    return args


def editable_config(
    transform_cfg: dict[str, Any], augmentation: str | None
) -> tuple[dict[str, Any], list[ParamRow]]:
    """The isolated config with the tested augmentation's blocks written out in full (the resolved values), and
    one ``ParamRow`` per numeric field of them (``prob`` excluded: it is 1)."""
    cfg = isolate(transform_cfg, augmentation)
    args = resolve(cfg)
    cfg["resize_interpolation"] = args.resize_interpolation
    cfg["resize_upscale_interpolation"] = args.resize_upscale_interpolation
    if augmentation is None:
        return cfg, []
    paths = [(augmentation,)]
    if augmentation in LOCAL_OVERRIDES:
        paths.append(("local_view", augmentation))
    rows = []
    for path in paths:
        block = _resolved_block(args, path).model_dump()
        _block(cfg, path[:-1])[path[-1]] = block
        for name, value in block.items():
            if name == "prob" or value is None or isinstance(value, (str, bool)):
                continue
            values = list(value) if isinstance(value, (tuple, list)) else [value]
            is_int = all(isinstance(v, int) for v in values)
            vmin, vmax = AUG_FIELD_RANGES.get(
                (augmentation, name), FIELD_RANGES.get(name, (0.0, 1.0))
            )
            vmin, vmax = min(vmin, *values), max(vmax, *values)
            indices = (
                [None]
                if len(values) == 1 and not isinstance(value, (tuple, list))
                else range(len(values))
            )
            rows += [ParamRow(path, name, i, vmin, vmax, is_int) for i in indices]
    return cfg, rows


def default_scales(
    transform_cfg: dict[str, Any],
) -> dict[str, tuple[float, tuple[float, float]]]:
    """``{view: (start scale, config range)}``: the middle of each view's ``random_resize`` range."""
    args = resolve(isolate(transform_cfg, None))
    ranges = {"global": args.random_resize, "local": args.local_view.random_resize}
    out = {}
    for name, r in ranges.items():
        lo, hi = (1.0, 1.0) if r is None else (r.min_scale, r.max_scale)
        out[name] = ((lo + hi) / 2, (lo, hi))
    return out


def view_index(index: int, start: int, crop: int, out: int) -> int | None:
    """The view index showing volume index ``index`` along one axis (``None`` if the crop doesn't contain it)."""
    if not start <= index < start + crop:
        return None
    return min(int((index - start + 0.5) * out / crop), out - 1)


#: The resampler's modes, one per axis group (in-plane, depth): ``RandomResizedCrop3D``'s interpolation specs.
RESIZE_MODES = tuple(_WEIGHT_FNS)
#: MONAI ``RandRotate``'s modes (``random_rotation.interpolation``).
ROTATION_MODES = ("bilinear", "nearest", "bicubic")
INTERPOLATION_KEYS = ("resize_interpolation", "resize_upscale_interpolation")


def interpolation_spec(in_plane: str, depth: str) -> str:
    """The config's spelling: ``"X"`` if both modes are ``X``, else ``"<in-plane>+<depth>"``."""
    return in_plane if in_plane == depth else f"{in_plane}+{depth}"


@dataclass
class Choice:
    """One radio group: an interpolation config field, or one axis group of it."""

    key: str  # an INTERPOLATION_KEYS entry, or "random_rotation"
    axis: int | None  # 0 in-plane, 1 depth; None for the rotation's single mode
    label: str
    options: tuple[str, ...]

    def get(self, cfg: dict[str, Any]) -> str:
        if self.axis is None:
            return cfg[self.key]["interpolation"]
        spec = cfg.get(self.key)
        return "off" if spec is None else parse_interpolation(spec)[self.axis]

    def set(self, cfg: dict[str, Any], value: str) -> None:
        if self.axis is None:
            cfg[self.key]["interpolation"] = value
            return
        if value == "off":
            cfg[self.key] = None  # only resize_upscale_interpolation offers it
            return
        spec = cfg.get(self.key)
        modes = [value, value] if spec is None else list(parse_interpolation(spec))
        modes[self.axis] = value
        cfg[self.key] = interpolation_spec(*modes)


def interpolation_choices(augmentation: str | None) -> list[Choice]:
    choices = [
        Choice("resize_interpolation", 0, "resize H/W", RESIZE_MODES),
        Choice("resize_interpolation", 1, "resize D", RESIZE_MODES),
        Choice(
            "resize_upscale_interpolation", 0, "upscale H/W", ("off", *RESIZE_MODES)
        ),
        Choice("resize_upscale_interpolation", 1, "upscale D", ("off", *RESIZE_MODES)),
    ]
    if augmentation == "random_rotation":
        choices.append(Choice("random_rotation", None, "rotation", ROTATION_MODES))
    return choices


# --------------------------------------------------------------------------- the viewer

ROW_H, ROW_GAP = 0.024, 0.008
SLIDER_X, SLIDER_W = 0.19, 0.36
TEXTBOX_X, TEXTBOX_W = 0.585, 0.07
RADIO_X, RADIO_RIGHT, RADIO_H = 0.68, 0.99, 0.15
CONTROLS_Y, CONTROLS_H = 0.02, 0.045
MIN_ROWS = 6  # room for the radio groups


class Viewer:
    def __init__(
        self,
        renderer: Renderer,
        transform_cfg: dict[str, Any],
        augmentation: str | None,
        seed: int = 0,
        title: str = "",
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import (
            Button,
            CheckButtons,
            RadioButtons,
            Slider,
            TextBox,
        )

        self._Slider, self._TextBox = Slider, TextBox
        self.renderer = renderer
        self.augmentation = augmentation
        self.seed = seed
        self.initial_cfg, self.rows = editable_config(transform_cfg, augmentation)
        self.cfg = copy.deepcopy(self.initial_cfg)
        self.original_cfg = isolate(transform_cfg, None)
        self.choices = interpolation_choices(augmentation)
        self._sync_interpolation()
        starts = default_scales(transform_cfg)
        self.scale_ranges = {name: r for name, (_, r) in starts.items()}
        self.initial_scales = {name: s for name, (s, _) in starts.items()}
        self.scales = dict(self.initial_scales)
        self.show_original = False
        self.physical_aspect = True
        d, h, w = renderer.volume.shape
        self.d_idx, self.w_idx = d // 2, w // 2
        self.error: str | None = None

        n_rows = max(4 + len(self.rows), MIN_ROWS)
        slider_top = CONTROLS_Y + CONTROLS_H + 0.02 + n_rows * (ROW_H + ROW_GAP)
        self.fig, axes = plt.subplots(2, 3, figsize=(15, 8.5 + 0.3 * n_rows))
        self.fig.subplots_adjust(
            left=0.04,
            right=0.98,
            top=0.88,
            bottom=slider_top + 0.04,
            hspace=0.45,
            wspace=0.15,
        )
        self.title = title
        self.axes = {"volume": axes[:, 0], "global": axes[:, 1], "local": axes[:, 2]}
        self.images: dict[str, list[Any]] = {}
        self.lines: dict[str, list[Any]] = {}
        for name, (ax_slice, ax_depth) in self.axes.items():
            self.images[name] = [
                ax_slice.imshow(
                    np.zeros((2, 2)),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                ),
                ax_depth.imshow(
                    np.zeros((2, 2)),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                ),
            ]
            # Slice row: where the depth row cuts (column W). Depth row: where the slice row cuts (slice D).
            self.lines[name] = [
                ax_slice.axvline(0, color="yellow", lw=0.8),
                ax_depth.axvline(0, color="yellow", lw=0.8),
            ]
            for ax in (ax_slice, ax_depth):
                ax.set_xticks([])
                ax.set_yticks([])
        self.boxes = {
            name: [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=False,
                    color="red",
                    lw=1.5,
                    ls="-" if name == "global" else "--",
                )
                for _ in range(2)
            ]
            for name in VIEWS
        }
        for name in VIEWS:
            for ax, box in zip(self.axes["volume"], self.boxes[name]):
                ax.add_patch(box)

        # Sliders: slice, column, the two crop scales, then the augmentation's fields.
        self.sliders: list[Any] = []
        self.textboxes: list[Any] = []
        y = slider_top - ROW_H
        specs: list[tuple[str, float, float, float, bool, Any]] = [
            ("slice D", 0, d - 1, self.d_idx, True, self._set_slice),
            ("column W", 0, w - 1, self.w_idx, True, self._set_column),
        ]
        for name in VIEWS:
            lo, hi = self.scale_ranges[name]
            specs.append(
                (
                    f"{name} crop scale (config {lo:g}-{hi:g})",
                    0.01,
                    1.0,
                    self.scales[name],
                    False,
                    self._scale_setter(name),
                )
            )
        for row in self.rows:
            specs.append(
                (
                    row.label,
                    row.vmin,
                    row.vmax,
                    row.get(self.cfg),
                    row.is_int,
                    self._param_setter(row),
                )
            )
        for label, vmin, vmax, value, is_int, callback in specs:
            self._add_row(y, label, vmin, vmax, value, is_int, callback)
            y -= ROW_H + ROW_GAP

        # Interpolation radio groups, side by side next to the sliders.
        self.radios: list[Any] = []
        width = (RADIO_RIGHT - RADIO_X) / len(self.choices)
        for i, choice in enumerate(self.choices):
            ax = self.fig.add_axes(
                (RADIO_X + i * width, slider_top - RADIO_H, width * 0.95, RADIO_H)
            )
            ax.set_title(choice.label, fontsize=8)
            radio = RadioButtons(
                ax, choice.options, active=choice.options.index(choice.get(self.cfg))
            )
            for text in radio.labels:
                text.set_fontsize(8)
            radio.on_clicked(self._choice_setter(choice))
            self.radios.append(radio)

        ax = self.fig.add_axes((0.04, CONTROLS_Y, 0.16, CONTROLS_H))
        self.check = CheckButtons(
            ax, ["Show original", "Physical aspect"], [False, True]
        )
        self.check.on_clicked(self._toggle)
        ax = self.fig.add_axes((0.62, CONTROLS_Y, 0.12, CONTROLS_H))
        self.resample_button = Button(ax, "Resample")
        self.resample_button.on_clicked(self._resample)
        ax = self.fig.add_axes((0.78, CONTROLS_Y, 0.12, CONTROLS_H))
        self.reset_button = Button(ax, "Reset")
        self.reset_button.on_clicked(self._reset)

        self.rendered: Rendered | None = None
        self.original: Rendered | None = None
        self.recompute()

    # --- widgets ------------------------------------------------------------

    def _add_row(
        self,
        y: float,
        label: str,
        vmin: float,
        vmax: float,
        value: float,
        is_int: bool,
        callback: Any,
    ) -> None:
        slider = self._Slider(
            self.fig.add_axes((SLIDER_X, y, SLIDER_W, ROW_H)),
            label,
            vmin,
            vmax,
            valinit=value,
            valstep=1 if is_int else None,
        )
        textbox = self._TextBox(
            self.fig.add_axes((TEXTBOX_X, y, TEXTBOX_W, ROW_H)),
            "",
            initial=_fmt(value, is_int),
        )

        def on_slider(val: float) -> None:
            _quiet_set(textbox, _fmt(val, is_int))
            callback(int(round(val)) if is_int else float(val))

        def on_submit(text: str) -> None:
            try:
                val = min(max(float(text), vmin), vmax)
            except ValueError:
                _quiet_set(textbox, _fmt(slider.val, is_int))
                return
            slider.set_val(round(val) if is_int else val)  # runs on_slider

        slider.on_changed(on_slider)
        textbox.on_submit(on_submit)
        self.sliders.append(slider)
        self.textboxes.append(textbox)

    def _set_slice(self, value: int) -> None:
        self.d_idx = value
        self.redraw()

    def _set_column(self, value: int) -> None:
        self.w_idx = value
        self.redraw()

    def _scale_setter(self, name: str) -> Any:
        def setter(value: float) -> None:
            self.scales[name] = value
            self.original = None
            self.recompute()

        return setter

    def _param_setter(self, row: ParamRow) -> Any:
        def setter(value: float) -> None:
            row.set(self.cfg, value)
            self.recompute()

        return setter

    def _choice_setter(self, choice: Choice) -> Any:
        def setter(value: str) -> None:
            choice.set(self.cfg, value)
            self._sync_interpolation()
            self.original = None
            self.recompute()

        return setter

    def _sync_interpolation(self) -> None:
        # The views without the augmentation are resized the same way.
        for key in INTERPOLATION_KEYS:
            self.original_cfg[key] = self.cfg[key]

    def _toggle(self, label: str | None) -> None:
        self.show_original, self.physical_aspect = self.check.get_status()
        if self.show_original and self.original is None:
            self.recompute()
        else:
            self.redraw()

    def _resample(self, _event: Any) -> None:
        self.seed += 1
        self.recompute()

    def _reset(self, _event: Any) -> None:
        self.cfg = copy.deepcopy(self.initial_cfg)
        self._sync_interpolation()
        self.scales = dict(self.initial_scales)
        self.original = None
        for radio, choice in zip(self.radios, self.choices):
            radio.eventson = False
            radio.set_active(choice.options.index(choice.get(self.cfg)))
            radio.eventson = True
        values = [
            self.d_idx,
            self.w_idx,
            *(self.scales[n] for n in VIEWS),
            *(r.get(self.cfg) for r in self.rows),
        ]
        for slider, textbox, value in zip(self.sliders, self.textboxes, values):
            slider.eventson = False
            slider.set_val(value)
            slider.eventson = True
            _quiet_set(textbox, _fmt(value, isinstance(value, int)))
        self.recompute()

    # --- rendering ----------------------------------------------------------

    def recompute(self) -> None:
        try:
            self.rendered = self.renderer.render(self.cfg, self.scales, self.seed)
            if self.show_original and self.original is None:
                self.original = self.renderer.render(
                    self.original_cfg, self.scales, self.seed
                )
            self.error = None
        except Exception as exc:  # noqa: BLE001 - an invalid setting is shown, the last good rendering kept
            self.error = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
        self.redraw()

    def redraw(self) -> None:
        rendered = self.rendered
        if rendered is None:
            self.fig.suptitle(f"{self.title}\n{self.error}", color="red")
            self.fig.canvas.draw_idle()
            return
        shown = (
            self.original
            if (self.show_original and self.original is not None)
            else rendered
        )
        d, h, w = rendered.volume.shape
        self._draw(
            "volume", rendered.volume, rendered.volume_spacing, self.d_idx, self.w_idx
        )
        self.axes["volume"][0].set_title(f"series {h}x{w}x{d}, slice D={self.d_idx}")
        self.axes["volume"][1].set_title(
            f"column W={self.w_idx} (red: global, dashed: local)"
        )
        for name in VIEWS:
            p = rendered.crops[name]
            out_d, _, out_w = shown.views[name].shape
            vd = view_index(self.d_idx, p.d_start, p.depth, out_d)
            vw = view_index(self.w_idx, p.w_start, p.width, out_w)
            self._draw(name, shown.views[name], rendered.spacings[name], vd, vw)
            missing = [a for a, v in (("slice", vd), ("column", vw)) if v is None]
            note = f" ({' and '.join(missing)} outside the crop)" if missing else ""
            label = (
                "original"
                if shown is not rendered
                else (self.augmentation or "no augmentation")
            )
            self.axes[name][0].set_title(
                f"{name} view, {label}{note}\n{rendered.info[name]}", fontsize=8
            )
            for box, (x0, xn, y0, yn) in zip(
                self.boxes[name],
                (
                    (p.w_start, p.width, p.h_start, p.height),
                    (p.d_start, p.depth, p.h_start, p.height),
                ),
            ):
                box.set_xy((x0 - 0.5, y0 - 0.5))
                box.set_width(xn)
                box.set_height(yn)
        upscale = self.cfg["resize_upscale_interpolation"] or "off"
        status = (
            f"resize {self.cfg['resize_interpolation']}, upscale {upscale}   |   seed {self.seed}"
            + (f"   |   {self.error}" if self.error else "")
        )
        self.fig.suptitle(
            f"{self.title}\n{status}",
            color="red" if self.error else "black",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def _draw(
        self,
        name: str,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        d: int | None,
        w: int | None,
    ) -> None:
        """Slice ``d`` (H x W) in the top row, column ``w`` (H x D) in the bottom one.

        ``None`` (outside a view's crop) takes the middle and hides the line marking it in the other row.
        """
        depth, height, width = volume.shape
        marks = (w, d)
        d = depth // 2 if d is None else d
        w = width // 2 if w is None else w
        sh, sw, sd = spacing if self.physical_aspect else (1.0, 1.0, 1.0)
        planes = (volume[d], volume[:, :, w].T)
        for ax, image, line, plane, aspect, x, mark in zip(
            self.axes[name],
            self.images[name],
            self.lines[name],
            planes,
            (sh / sw, sh / sd),
            (w, d),
            marks,
        ):
            line.set_visible(mark is not None)
            image.set_data(np.clip(plane, 0.0, 1.0))
            image.set_extent((-0.5, plane.shape[1] - 0.5, plane.shape[0] - 0.5, -0.5))
            ax.set_xlim(-0.5, plane.shape[1] - 0.5)
            ax.set_ylim(plane.shape[0] - 0.5, -0.5)
            ax.set_aspect(aspect)
            line.set_xdata([x, x])


def _fmt(value: float, is_int: bool) -> str:
    return str(int(round(value))) if is_int else f"{value:.4g}"


def _quiet_set(textbox: Any, text: str) -> None:
    # TextBox.set_val fires its submit event, which would feed the rounded text back into the slider.
    textbox.eventson = False
    try:
        textbox.set_val(text)
    finally:
        textbox.eventson = True


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "series_dir",
        type=Path,
        help="A series folder in the KneeNo layout (000.jpeg, 001.jpeg, ...)",
    )
    parser.add_argument(
        "augmentation",
        choices=[*AUGMENTATIONS, "none"],
        help="The augmentation to test ('none': crop/resize only)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("H", "W", "D"),
        default=None,
        help="Voxel spacing in mm, [H, W, D] like the metadata's data_spacing (default: none, the 1/shape fallback)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Run config whose transform: block (and data.series_depth / resample_mode) to use",
    )
    parser.add_argument(
        "--series-depth",
        type=int,
        default=None,
        help="Default: the config's data.series_depth",
    )
    parser.add_argument(
        "--resample-mode",
        choices=["nearest", "interpolate"],
        default=None,
        help="Default: the config's data.resample_mode",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the augmentation's random draws"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.series_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.series_dir}")
    if args.spacing is not None and not all(s > 0 for s in args.spacing):
        raise SystemExit(f"--spacing must be positive, got {args.spacing}")
    transform_cfg, data_cfg = load_config(args.config)
    series_depth = (
        args.series_depth
        if args.series_depth is not None
        else int(data_cfg.get("series_depth", 0) or 0)
    )
    resample_mode = args.resample_mode or data_cfg.get("resample_mode", "nearest")
    augmentation = None if args.augmentation == "none" else args.augmentation

    dataset, _tmp = load_series(
        args.series_dir, args.spacing, series_depth, resample_mode
    )
    renderer = Renderer(dataset, has_spacing=args.spacing is not None)
    spacing = "spacing " + (
        "x".join(f"{s:g}" for s in args.spacing) + " mm"
        if args.spacing
        else "none (1/shape)"
    )
    title = (
        f"{args.series_dir.parent.name}/{args.series_dir.name}: {args.augmentation}   |   {spacing}, "
        f"series_depth {series_depth or 'native'} ({resample_mode})   |   {args.config.name}"
    )
    Viewer(renderer, transform_cfg, augmentation, seed=args.seed, title=title)
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    sys.exit(main())
