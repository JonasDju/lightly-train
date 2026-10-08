#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
import torch
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.adapter import EncoderAdapter

from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._embedding.embedding_transform import EmbeddingTransform

from .. import helpers


def _adapter(dataset_type: str = "internal") -> DINOv2Adapter:
    return DINOv2Adapter(
        dataset_type=dataset_type, embed_dim=8, num_channels=3, image_size=(32, 16)
    )


def test_is_encoder_adapter() -> None:
    assert isinstance(_adapter(), EncoderAdapter)


def test_has_cls_token() -> None:
    # DINOv2 has a cls token, so KneeNo's "linear" task is available.
    assert DINOv2Adapter.has_cls_token is True


def test_init() -> None:
    adapter = DINOv2Adapter(
        dataset_type="external",
        embed_dim=8,
        num_channels=1,
        image_size=[32, 16],
        normalize=((0.5,), (0.25,)),
    )
    assert adapter.embed_dim == 8
    assert adapter.num_channels == 1
    assert adapter.dataset_type == "external"
    assert adapter.image_size == (32, 16)
    assert adapter.mean == (0.5,)
    assert adapter.std == (0.25,)


def test_init__invalid_dataset_type() -> None:
    with pytest.raises(ValueError, match="dataset_type"):
        _adapter(dataset_type="labeled")


def test_init__invalid_num_channels() -> None:
    with pytest.raises(ValueError, match="num_channels"):
        DINOv2Adapter(
            dataset_type="internal", embed_dim=8, num_channels=0, image_size=(32, 16)
        )


@pytest.mark.parametrize(
    "num_channels, normalize",
    [
        (3, ((0.5,), (0.25,))),  # one shared (mean, std)
        (3, ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))),  # one per channel
        (1, ((0.5,), (0.25,))),
    ],
)
def test_init__normalize_matches_num_channels(
    num_channels: int, normalize: tuple[tuple[float, ...], tuple[float, ...]]
) -> None:
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        num_channels=num_channels,
        image_size=(32, 16),
        normalize=normalize,
    )
    assert adapter.mean == normalize[0]
    assert adapter.std == normalize[1]


@pytest.mark.parametrize(
    "num_channels, normalize",
    [
        (1, ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))),  # RGB normalize for a 1-channel model
        (3, ((0.1, 0.2), (0.4, 0.5))),
        (3, ((0.1, 0.2, 0.3), (0.4,))),  # mean/std length mismatch
    ],
)
def test_init__normalize_mismatch(
    num_channels: int, normalize: tuple[tuple[float, ...], tuple[float, ...]]
) -> None:
    with pytest.raises(ValueError, match="normalize"):
        DINOv2Adapter(
            dataset_type="internal",
            embed_dim=8,
            num_channels=num_channels,
            image_size=(32, 16),
            normalize=normalize,
        )


def test_picklable() -> None:
    # prepare_input runs in DataLoader workers, which may be spawned.
    adapter = _adapter()
    unpickled = pickle.loads(pickle.dumps(adapter))
    assert unpickled.embed_dim == 8
    volume = _random_volume(depth=3)
    assert torch.equal(unpickled.prepare_input(volume), adapter.prepare_input(volume))


# -- prepare_input -------------------------------------------------------------------


def _random_volume(
    depth: int = 4, height: int = 20, width: int = 11, seed: int = 0
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (1, depth, height, width), generator=gen, dtype=torch.uint8
    )


def _expected_constant(
    value: float, mean: tuple[float, ...], std: tuple[float, ...], num_channels: int
) -> torch.Tensor:
    """Normalized per-channel value of a constant slice (resizing keeps it constant)."""
    mean_t = torch.tensor(mean).expand(num_channels)
    std_t = torch.tensor(std).expand(num_channels)
    return (value / 255.0 - mean_t) / std_t


@pytest.mark.parametrize("num_channels", [1, 3])
def test_prepare_input__shape_and_dtype(num_channels: int) -> None:
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        num_channels=num_channels,
        image_size=(32, 16),
        normalize=((0.5,), (0.25,)),
    )
    out = adapter.prepare_input(_random_volume(depth=5, height=20, width=11))
    # (C, D, image_size[0], image_size[1]): every slice is resized, depth is kept.
    assert out.shape == (num_channels, 5, 32, 16)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    "num_channels, normalize",
    [
        (3, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),  # stock ImageNet
        (3, ((0.5,), (0.25,))),  # one shared (mean, std)
        (1, ((0.5,), (0.25,))),
    ],
)
def test_prepare_input__normalizes_each_slice_in_order(
    num_channels: int, normalize: tuple[tuple[float, ...], tuple[float, ...]]
) -> None:
    # Slice d is constant with value 40 * d, so each output slice is constant too and
    # its value shows which input slice it came from and how it was normalized.
    values = [0, 40, 80, 120, 255]
    volume = torch.stack([torch.full((20, 11), v, dtype=torch.uint8) for v in values])[
        None
    ]
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        num_channels=num_channels,
        image_size=(16, 16),
        normalize=normalize,
    )

    out = adapter.prepare_input(volume)

    for d, value in enumerate(values):
        expected = _expected_constant(value, *normalize, num_channels=num_channels)
        torch.testing.assert_close(
            out[:, d], expected.view(-1, 1, 1).expand(-1, 16, 16)
        )


def test_prepare_input__keeps_spatial_layout() -> None:
    # Height and width must not be swapped: a bright right half stays on the right.
    volume = torch.zeros(1, 2, 20, 12, dtype=torch.uint8)
    volume[..., 6:] = 255
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        num_channels=1,
        image_size=(16, 8),
        normalize=((0.0,), (1.0,)),
    )

    out = adapter.prepare_input(volume)

    assert out.shape == (1, 2, 16, 8)
    torch.testing.assert_close(out[..., :3], torch.zeros(1, 2, 16, 3))
    torch.testing.assert_close(out[..., 5:], torch.ones(1, 2, 16, 3))


def test_prepare_input__slices_are_independent() -> None:
    adapter = _adapter()
    volume = _random_volume(depth=4)
    changed = volume.clone()
    changed[:, 2] = 255 - changed[:, 2]

    out, out_changed = adapter.prepare_input(volume), adapter.prepare_input(changed)

    assert not torch.equal(out[:, 2], out_changed[:, 2])
    for d in (0, 1, 3):
        assert torch.equal(out[:, d], out_changed[:, d])


def test_prepare_input__matches_embedding_transform() -> None:
    # Each slice goes through lightly-train's EmbeddingTransform as an RGB image.
    adapter = _adapter()
    volume = _random_volume(depth=3)
    transform = EmbeddingTransform(
        image_size=adapter.image_size, mean=adapter.mean, std=adapter.std
    )

    out = adapter.prepare_input(volume)

    for d in range(3):
        image = np.repeat(volume[0, d].numpy()[..., None], 3, axis=-1)  # (H, W, 3)
        expected = transform({"image": image})[0]["image"]
        torch.testing.assert_close(out[:, d], expected)


def test_prepare_input__float_volume() -> None:
    # Resampling with resample_mode="interpolate" yields float32 volumes in 0..255.
    values = [0, 100, 255]
    volume = torch.stack([torch.full((20, 11), float(v)) for v in values])[None]
    adapter = _adapter()

    out = adapter.prepare_input(volume)

    for d, value in enumerate(values):
        expected = _expected_constant(value, adapter.mean, adapter.std, 3)
        torch.testing.assert_close(
            out[:, d], expected.view(-1, 1, 1).expand(-1, 32, 16)
        )


@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("dtype", [torch.int16, torch.float32])
def test_prepare_input__external_is_rescaled_to_uint8(
    num_channels: int, dtype: torch.dtype
) -> None:
    # External volumes come back raw (intensities far outside 0..255); after
    # LabeledExternalKneeMRIDataset.to_uint8 they are handled like internal ones.
    gen = torch.Generator().manual_seed(0)
    volume = torch.randint(0, 4000, (1, 4, 20, 11), generator=gen).to(dtype)
    kwargs: dict[str, Any] = dict(
        embed_dim=8,
        num_channels=num_channels,
        image_size=(32, 16),
        normalize=((0.5,), (0.25,)),
    )
    external = DINOv2Adapter(dataset_type="external", **kwargs)
    internal = DINOv2Adapter(dataset_type="internal", **kwargs)

    out = external.prepare_input(volume)

    expected = internal.prepare_input(
        torch.from_numpy(LabeledExternalKneeMRIDataset.to_uint8(volume))
    )
    assert out.shape == (num_channels, 4, 32, 16)
    torch.testing.assert_close(out, expected)


def test_prepare_input__internal_is_not_rescaled() -> None:
    # Internal volumes are already 0..255: a dark volume must stay dark, not be
    # stretched to the full range like an external one.
    volume = torch.full((1, 2, 20, 11), 10, dtype=torch.uint8)
    volume[..., 0, 0] = 20
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        num_channels=1,
        image_size=(20, 11),
        normalize=((0.0,), (1.0,)),
    )

    out = adapter.prepare_input(volume)

    assert out.max().item() == pytest.approx(20 / 255)


def test_prepare_input__ignores_orientation() -> None:
    adapter = _adapter()
    volume = _random_volume()
    assert torch.equal(
        adapter.prepare_input(volume, orientation="LIP"), adapter.prepare_input(volume)
    )


def test_prepare_input__multi_channel_volume_raises() -> None:
    volume = torch.zeros(3, 4, 20, 11, dtype=torch.uint8)
    with pytest.raises(ValueError, match="1-channel volume"):
        _adapter().prepare_input(volume)


# -- forward_features ----------------------------------------------------------------

_IMAGE_SIZE = 16
_PATCH_SIZE = 2
_NUM_PATCHES = (_IMAGE_SIZE // _PATCH_SIZE) ** 2


def _model() -> Any:
    torch.manual_seed(0)
    return helpers.dummy_dinov2_vit_model(
        patch_size=_PATCH_SIZE, img_size=_IMAGE_SIZE
    ).eval()


def _model_adapter(model: Any, num_channels: int = 3) -> DINOv2Adapter:
    return DINOv2Adapter(
        dataset_type="internal",
        embed_dim=model.feature_dim(),
        num_channels=num_channels,
        image_size=(_IMAGE_SIZE, _IMAGE_SIZE),
    )


def test_forward_features__shapes() -> None:
    model = _model()
    embed_dim = model.feature_dim()
    batch = torch.randn(2, 3, 5, _IMAGE_SIZE, _IMAGE_SIZE)

    with torch.no_grad():
        out = _model_adapter(model).forward_features(model, batch)

    assert set(out) == {"cls", "patches"}
    cls, patches = out["cls"], out["patches"]
    assert cls is not None and patches is not None
    assert cls.shape == (2, embed_dim)
    # All slices' patch tokens: D * P per volume.
    assert patches.shape == (2, 5 * _NUM_PATCHES, embed_dim)


def test_forward_features__matches_per_slice_encoding() -> None:
    # cls = mean of the slices' cls tokens, patches = the slices' patch tokens in
    # slice order, and volumes in a batch do not mix.
    model = _model()
    batch = torch.randn(2, 3, 4, _IMAGE_SIZE, _IMAGE_SIZE)

    with torch.no_grad():
        out = _model_adapter(model).forward_features(model, batch)
        cls, patches = out["cls"], out["patches"]
        assert cls is not None and patches is not None
        for b in range(2):
            slices = [model.forward_features(batch[b, :, d][None]) for d in range(4)]
            expected_cls = torch.stack([s["cls_token"][0] for s in slices]).mean(0)
            # (E, h, w) -> (h * w, E) per slice, concatenated over slices.
            expected_patches = torch.cat(
                [s["features"][0].flatten(1).T for s in slices]
            )
            torch.testing.assert_close(cls[b], expected_cls)
            torch.testing.assert_close(patches[b], expected_patches)


def test_forward_features__unwraps_ddp() -> None:
    class _DDPLike(torch.nn.Module):
        # Like DistributedDataParallel: the model sits in .module and the wrapper
        # itself has no forward_features.
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

    model = _model()
    adapter = _model_adapter(model)
    batch = torch.randn(2, 3, 2, _IMAGE_SIZE, _IMAGE_SIZE)

    with torch.no_grad():
        out = adapter.forward_features(_DDPLike(model), batch)
        expected = adapter.forward_features(model, batch)

    torch.testing.assert_close(out["cls"], expected["cls"])
    torch.testing.assert_close(out["patches"], expected["patches"])


@pytest.mark.parametrize("dataset_type", ["internal", "external"])
def test_prepare_input__collate__forward_features(dataset_type: str) -> None:
    # The path KneeNo takes: prepare_input per volume (in DataLoader workers), the
    # default collate, then forward_features on the batch.
    model = _model()
    adapter = DINOv2Adapter(
        dataset_type=dataset_type,
        embed_dim=model.feature_dim(),
        num_channels=3,
        image_size=(_IMAGE_SIZE, _IMAGE_SIZE),
    )
    volumes = [_random_volume(depth=3, seed=seed) for seed in range(4)]

    batch = adapter.collate([adapter.prepare_input(v) for v in volumes])
    with torch.no_grad():
        out = adapter.forward_features(model, batch)

    assert batch.shape == (4, 3, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    cls, patches = out["cls"], out["patches"]
    assert cls is not None and patches is not None
    assert cls.shape == (4, model.feature_dim())
    assert patches.shape == (4, 3 * _NUM_PATCHES, model.feature_dim())
    assert torch.isfinite(cls).all()
    assert torch.isfinite(patches).all()
