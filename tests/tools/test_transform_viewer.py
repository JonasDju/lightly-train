#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""``tools/transform_viewer.py``: the viewer must show what training does, apart from its fixed crop."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from lightly_train._commands import train_helpers
from lightly_train._data.mi_dataset import reseed_randomizables
from lightly_train._transforms.monai_wrappers import (
    VOXEL_SPACING_META_KEY,
    AnisotropyAwareRandGaussianSmooth,
    AnisotropyTrackingRandomResizedCrop3D,
)

from .. import helpers

matplotlib.use("Agg")

_PATH = Path(__file__).resolve().parents[2] / "tools" / "transform_viewer.py"
_SPEC = importlib.util.spec_from_file_location("transform_viewer", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
tv = importlib.util.module_from_spec(_SPEC)
sys.modules["transform_viewer"] = tv
_SPEC.loader.exec_module(tv)

SPACING = (0.3, 0.35, 3.0)  # [H, W, D] mm, distinct so that an axis mix-up shows


def _config() -> dict[str, Any]:
    transform, _ = tv.load_config(tv.DEFAULT_CONFIG)
    # Small views keep the tests fast; non-cubic so that an axis mix-up shows.
    transform["image_size"] = [40, 36, 8]
    transform["local_view"] = {
        **(transform.get("local_view") or {}),
        "view_size": [20, 18, 4],
    }
    return transform


def _series(tmp_path: Path) -> Path:
    data_root, _ = helpers.create_mi_dataset(
        tmp_path, n_cases=1, depths=(9,), height=48, width=44
    )
    return Path(data_root["internal"]) / "case_0" / "series_0"


def _renderer(tmp_path: Path, spacing: Any = SPACING) -> Any:
    dataset, _ = tv.load_series(
        _series(tmp_path), spacing, series_depth=12, resample_mode="nearest"
    )
    return tv.Renderer(dataset, has_spacing=spacing is not None)


SCALES = {"global": 0.5, "local": 0.1}


@pytest.mark.parametrize("scale", [0.02, 0.1, 0.32, 0.5, 0.66, 1.0])
def test_center_crop__training_sizes_centred(scale: float) -> None:
    shape = (1, 256, 224, 20)
    real = AnisotropyTrackingRandomResizedCrop3D(
        size=(16, 16, 8), scale=(scale, scale), ratio=(1.0, 1.0), seed=0
    )
    crop = tv.CenterCrop3D(real, scale)
    params = crop.get_params(shape)
    drawn = real.get_params(
        shape
    )  # the training draw for this scale and no aspect jitter
    assert (params.height, params.width, params.depth) == (
        drawn.height,
        drawn.width,
        drawn.depth,
    )
    assert (params.h_start, params.w_start, params.d_start) == (
        (256 - params.height) // 2,
        (224 - params.width) // 2,
        (20 - params.depth) // 2,
    )


def test_center_crop__tags_the_view_spacing() -> None:
    real = AnisotropyTrackingRandomResizedCrop3D(
        size=(16, 12, 4),
        interpolation="area+nearest",
        upscale_interpolation="linear",
        output_dtype=np.float32,
    )
    crop = tv.CenterCrop3D(real, 0.5)
    volume = MetaTensor(
        torch.rand(1, 40, 36, 12), meta={VOXEL_SPACING_META_KEY: SPACING}
    )
    out = crop(volume)
    p = crop.last_params
    expected = (0.3 * p.height / 16, 0.35 * p.width / 12, 3.0 * p.depth / 4)
    assert crop.last_spacing == pytest.approx(expected)
    assert out.meta[VOXEL_SPACING_META_KEY] == pytest.approx(expected)
    for attribute in ("size", "interpolation", "upscale_interpolation", "output_dtype"):
        assert getattr(crop, attribute) == getattr(real, attribute)


def test_isolate__only_the_tested_augmentation() -> None:
    config = _config()
    before = copy.deepcopy(config)
    cfg = tv.isolate(config, "gaussian_blur")
    assert config == before  # not mutated
    assert cfg["gaussian_blur"]["prob"] == 1.0
    assert cfg["gaussian_blur"]["sigma_range"] == config["gaussian_blur"]["sigma_range"]
    assert all(
        cfg[name] is None for name in tv.AUGMENTATIONS if name != "gaussian_blur"
    )
    assert all(value is None for value in cfg["global_view_1"].values())
    assert cfg["local_view"]["num_views"] == 1
    assert cfg["local_view"]["gaussian_blur"]["prob"] == 1.0
    assert cfg["local_view"]["gaussian_sharpen"] is None
    assert cfg["crop_foreground"] is None
    # An augmentation the config leaves off starts from the args' defaults.
    assert tv.isolate(config, "random_rotation")["random_rotation"] == {
        "degrees": 15.0,
        "prob": 1.0,
    }
    off = tv.isolate(config, None)
    assert all(off[name] is None for name in tv.AUGMENTATIONS)
    assert off["local_view"]["gaussian_blur"] is None


def test_editable_config__rows() -> None:
    cfg, rows = tv.editable_config(_config(), "gaussian_blur")
    assert [row.label for row in rows] == [
        "sigma_range lo",
        "sigma_range hi",
        "local sigma_range lo",
        "local sigma_range hi",
    ]
    rows[3].set(cfg, 0.9)
    assert cfg["local_view"]["gaussian_blur"]["sigma_range"][1] == 0.9
    assert rows[3].get(cfg) == 0.9
    _, rows = tv.editable_config(_config(), "histogram_shift")
    assert [(row.label, row.is_int) for row in rows] == [
        ("alpha", False),
        ("num_control_points", True),
    ]


def test_interpolation_choices() -> None:
    cfg, _ = tv.editable_config(_config(), "random_rotation")
    # Written out from the config (area, upscale linear).
    assert (cfg["resize_interpolation"], cfg["resize_upscale_interpolation"]) == (
        "area",
        "linear",
    )
    resize_hw, resize_d, upscale_hw, upscale_d, rotation = tv.interpolation_choices(
        "random_rotation"
    )
    resize_d.set(cfg, "nearest")
    assert cfg["resize_interpolation"] == "area+nearest"
    assert (resize_hw.get(cfg), resize_d.get(cfg)) == ("area", "nearest")
    resize_hw.set(cfg, "nearest")
    assert cfg["resize_interpolation"] == "nearest"  # the config's short form
    upscale_d.set(cfg, "off")
    assert cfg["resize_upscale_interpolation"] is None
    assert upscale_hw.get(cfg) == upscale_d.get(cfg) == "off"
    upscale_d.set(cfg, "cubic")
    assert cfg["resize_upscale_interpolation"] == "cubic"
    upscale_hw.set(cfg, "linear")
    assert cfg["resize_upscale_interpolation"] == "linear+cubic"
    rotation.set(cfg, "nearest")
    assert cfg["random_rotation"]["interpolation"] == "nearest"
    tv.resolve(cfg)  # all of it valid config
    assert tv.RESIZE_MODES == ("area", "linear", "nearest", "cubic")
    assert [c.label for c in tv.interpolation_choices(None)] == [
        "resize H/W",
        "resize D",
        "upscale H/W",
        "upscale D",
    ]


def test_series_spacing_is_the_training_one(tmp_path: Path) -> None:
    renderer = _renderer(tmp_path)
    # 9 slices resampled to 12: the slice step shrinks by 8 / 11.
    assert renderer.volume.shape == (12, 48, 44)
    assert renderer.volume_spacing == pytest.approx((0.3, 0.35, 3.0 * 8 / 11))
    fallback = _renderer(tmp_path / "fallback", spacing=None)
    assert fallback.volume_spacing == pytest.approx((1 / 48, 1 / 44, 1 / 12))


@pytest.mark.parametrize("spacing", [SPACING, None])
def test_render__blur_scaled_by_the_view_spacing(tmp_path: Path, spacing: Any) -> None:
    renderer = _renderer(tmp_path, spacing)
    cfg, _ = tv.editable_config(_config(), "gaussian_blur")
    rendered = renderer.render(cfg, SCALES, seed=0)
    transform, views = tv.build_transform(
        cfg, SCALES
    )  # render's construction, kept to inspect the blur
    for name, size in (("global", (40, 36, 8)), ("local", (20, 18, 4))):
        p = rendered.crops[name]
        h, w, d = rendered.spacings[name]
        assert (h, w, d) == pytest.approx(
            tuple(
                s * c / o
                for s, c, o in zip(
                    renderer.volume_spacing, (p.height, p.width, p.depth), size
                )
            )
        )
        assert rendered.views[name].shape == (size[2], size[0], size[1])
        assert "scale_z" in rendered.info[name] and "sigma x" in rendered.info[name]
    # The blur's depth range is the configured one times sqrt(h * w) / d of the view spacing.
    renderer.dataset.transform = transform
    reseed_randomizables(transform, rng=np.random.RandomState(0))
    renderer.dataset[0]
    blur = next(
        s
        for s in views["global"].transform.transforms
        if isinstance(s, AnisotropyAwareRandGaussianSmooth)
    )
    h, w, d = rendered.spacings["global"]
    lo, hi = cfg["gaussian_blur"]["sigma_range"]
    assert blur.sigma_z == pytest.approx(
        (lo * math.sqrt(h * w) / d, hi * math.sqrt(h * w) / d)
    )


@pytest.mark.parametrize(
    "scales",
    [
        SCALES,
        # Crops smaller than the views on every axis: the resize enlarges (resize_upscale_interpolation).
        {"global": 0.3, "local": 0.02},
    ],
)
@pytest.mark.parametrize(
    "interpolation",
    [None, ("cubic+nearest", None), ("nearest", "cubic+linear")],
)
@pytest.mark.parametrize(
    "augmentation", [None, "gaussian_blur", "gaussian_sharpen", "random_rotation"]
)
def test_render__equals_training_with_the_same_crops(
    tmp_path: Path,
    augmentation: str | None,
    scales: dict[str, float],
    interpolation: tuple[str, str | None] | None,
) -> None:
    """The viewer's views are the training transform's for the same crop: built independently here, with its
    crops forced to the viewer's crop parameters and everything else (resampling, tags, blur, ...) its own."""
    renderer = _renderer(tmp_path)
    cfg, _ = tv.editable_config(_config(), augmentation)
    if interpolation is not None:
        cfg["resize_interpolation"], cfg["resize_upscale_interpolation"] = interpolation
    rendered = renderer.render(cfg, scales, seed=3)

    transform = train_helpers.get_transform(
        method="dinov2", transform_args_resolved=tv.resolve(cfg)
    )
    transform.transforms = [transform.transforms[0], transform.transforms[2]]
    for view, name in zip(transform.transforms, ("global", "local")):
        crop = view.transform.transforms[0]
        assert type(crop) is AnisotropyTrackingRandomResizedCrop3D
        crop.get_params = lambda shape, foreground_box=None, p=rendered.crops[name]: p
    reseed_randomizables(transform, rng=np.random.RandomState(3))
    torch.manual_seed(3)
    renderer.dataset.transform = transform
    outputs = renderer.dataset[0]["views"]
    mean, std = (
        transform.transform_args.normalize.mean[0],
        transform.transform_args.normalize.std[0],
    )
    for name, output in zip(("global", "local"), outputs):
        np.testing.assert_array_equal(
            rendered.views[name], output[0].numpy() * std + mean
        )


def test_render__gaussian_noise_shows_the_drawn_strength(tmp_path: Path) -> None:
    """MONAI draws the noise std per call, uniform in [0, std]: two seeds give two strengths, both shown."""
    renderer = _renderer(tmp_path)
    cfg, _ = tv.editable_config(_config(), "gaussian_noise")
    cfg["gaussian_noise"]["std"] = 20.0
    drawn = []
    for seed in (0, 1):
        info = renderer.render(cfg, SCALES, seed=seed).info["global"]
        assert "(drawn from 0-20)" in info
        drawn.append(float(info.split("noise std ")[1].split()[0]))
    assert all(0 < value <= 20 for value in drawn)
    assert drawn[0] != pytest.approx(drawn[1], rel=0.05)


@pytest.mark.parametrize(
    "index, start, crop, out, expected",
    [
        # Volume indices 5..12 resized to 8: one to one.
        (4, 5, 8, 8, None),
        (5, 5, 8, 8, 0),
        (9, 5, 8, 8, 4),
        (12, 5, 8, 8, 7),
        (13, 5, 8, 8, None),
        # 16 voxels resized to 8: two per view voxel.
        (2, 0, 16, 8, 1),
        (3, 0, 16, 8, 1),
        (15, 0, 16, 8, 7),
    ],
)
def test_view_index(
    index: int, start: int, crop: int, out: int, expected: int | None
) -> None:
    assert tv.view_index(index, start=start, crop=crop, out=out) == expected


def test_viewer__widgets_and_errors(tmp_path: Path) -> None:
    renderer = _renderer(tmp_path)
    viewer = tv.Viewer(renderer, _config(), "histogram_shift", title="test")
    assert viewer.error is None
    first = viewer.rendered.views["global"].copy()
    viewer.sliders[2].set_val(0.9)  # global crop scale
    assert viewer.rendered.crops["global"].height > 0 and viewer.error is None
    viewer.sliders[4].set_val(0.9)  # alpha
    viewer._resample(None)
    viewer.check.set_active(0)  # show original
    assert viewer.original is not None
    viewer.check.set_active(1)  # voxel aspect
    viewer.sliders[0].set_val(0)
    viewer.sliders[1].set_val(0)
    # Interpolation: the augmented and the original views are resized alike.
    viewer.radios[1].set_active(tv.RESIZE_MODES.index("nearest"))  # resize D
    viewer.radios[2].set_active(0)  # upscale H/W: off
    assert viewer.error is None
    assert (
        viewer.cfg["resize_interpolation"]
        == viewer.original_cfg["resize_interpolation"]
        == "area+nearest"
    )
    assert (
        viewer.cfg["resize_upscale_interpolation"]
        is viewer.original_cfg["resize_upscale_interpolation"]
        is None
    )
    viewer._reset(None)
    assert (
        viewer.cfg["resize_interpolation"]
        == viewer.original_cfg["resize_interpolation"]
        == "area"
    )
    assert viewer.radios[1].value_selected == "area"
    np.testing.assert_array_equal(
        viewer.scales["global"], viewer.initial_scales["global"]
    )
    # An invalid setting is reported and the last good rendering kept.
    good = viewer.rendered
    viewer.rows[0].set(viewer.cfg, -1.0)  # alpha must be >= 0
    viewer.recompute()
    assert viewer.error is not None and viewer.rendered is good
    assert first.shape == good.views["global"].shape


def test_viewer__window_resize(tmp_path: Path) -> None:
    """matplotlib's own TextBox raised (headless) or printed a traceback (GUI) on every window resize."""
    from matplotlib.backend_bases import ResizeEvent

    viewer = tv.Viewer(_renderer(tmp_path), _config(), "gaussian_blur", title="test")
    assert all(isinstance(textbox, tv._TextBox) for textbox in viewer.textboxes)
    ResizeEvent(
        "resize_event", viewer.fig.canvas
    )._process()  # headless, a failing callback raises


def test_main__runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    tv.main(
        [
            str(_series(tmp_path)),
            "gaussian_blur",
            "--spacing",
            "0.3",
            "0.35",
            "3.0",
            "--series-depth",
            "12",
        ]
    )
    assert len(shown) == 1
