#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.adapter import EncoderAdapter

from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._transforms.random_resized_crop import (
    InterpolationMode,
    RandomResizedCrop3D,
)

from .. import helpers

# Non-cubic on purpose: H != W != D is what catches an (H, W, D) <-> (D, H, W) swap,
# which is invisible on cubic fixtures. See CLAUDE.md's axis-order gotcha.
IMAGE_SIZE = (16, 8, 4)  # (H, W, D)
PATCH_SIZE = (2, 2, 2)


def _adapter_and_model(
    dataset_type: str = "internal",
) -> tuple[DINOv2Adapter, DINOv2ViTModelWrapper]:
    wrapper = helpers.dummy_dinov2_vit_model(patch_size=PATCH_SIZE, img_size=IMAGE_SIZE)
    adapter = DINOv2Adapter(
        dataset_type=dataset_type,
        embed_dim=wrapper.feature_dim(),
        image_size=IMAGE_SIZE,
    )
    return adapter, wrapper


def test_dinov2_adapter__is_encoder_adapter() -> None:
    adapter, _ = _adapter_and_model()
    assert isinstance(adapter, EncoderAdapter)
    # DINOv2 has a cls token, unlike V-JEPA 2.1 -- this is what makes KneeNo's "linear"
    # task available for this model.
    assert adapter.has_cls_token is True


def test_dinov2_adapter__prepare_input_resizes_to_image_size() -> None:
    adapter, _ = _adapter_and_model()
    # Native volume shape (1, D, H, W), deliberately unlike the target size.
    volume = torch.randint(0, 256, (1, 7, 20, 11), dtype=torch.uint8)
    out = adapter.prepare_input(volume)
    # (H, W, D) config -> (D, H, W) tensor.
    assert out.shape == (1, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1])
    assert out.dtype == torch.float32


def test_dinov2_adapter__prepare_input_normalizes_like_view_transform() -> None:
    """Matches NormalizeIntensity(subtrahend=[m * 255], divisor=[s * 255])."""
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        image_size=IMAGE_SIZE,
        normalize=((0.25,), (0.5,)),
    )
    # Already the target size, so no interpolation runs and values map exactly.
    volume = torch.full((1, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1]), 255.0)
    out = adapter.prepare_input(volume)
    expected = (255.0 - 0.25 * 255.0) / (0.5 * 255.0)
    assert torch.allclose(out, torch.full_like(out, expected))


@pytest.mark.parametrize(
    "native_dhw",
    [
        (7, 20, 11),  # every axis shrinks
        (3, 20, 5),  # D and W grow (upscale mode), H shrinks
    ],
)
@pytest.mark.parametrize(
    "interpolation, upscale_interpolation",
    [
        ("area", "linear"),  # the DINOTransformArgs defaults
        ("area", None),  # area on every axis
        ("cubic", "nearest"),
    ],
)
def test_dinov2_adapter__prepare_input_resizes_like_training(
    native_dhw: tuple[int, int, int],
    interpolation: InterpolationMode,
    upscale_interpolation: InterpolationMode | None,
) -> None:
    """Same resampler as training: RandomResizedCrop3D at scale=ratio=1 with the run's
    resize_interpolation / resize_upscale_interpolation, then normalize.

    A trilinear resize instead keeps noise and aliasing on downsampled axes that the
    area-resampled training views never contain.
    """
    adapter = DINOv2Adapter(
        dataset_type="internal",
        embed_dim=8,
        image_size=IMAGE_SIZE,
        resize_interpolation=interpolation,
        resize_upscale_interpolation=upscale_interpolation,
    )
    volume = torch.randint(0, 256, (1, *native_dhw), dtype=torch.uint8)

    crop = RandomResizedCrop3D(
        size=IMAGE_SIZE,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
        interpolation=interpolation,
        upscale_interpolation=upscale_interpolation,
        output_dtype=np.float32,
    )
    resized = crop(volume.permute(0, 2, 3, 1).numpy())  # (1, H, W, D)
    expected = (torch.from_numpy(resized).permute(0, 3, 1, 2) - 127.5) / 127.5

    out = adapter.prepare_input(volume)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32])
def test_dinov2_adapter__external_volume_is_prepared_like_its_uint8_version(
    dtype: type,
) -> None:
    """dataset_type="external": the raw NIfTI volume is first quantized the way the internal
    JPEGs were (LabeledExternalKneeMRIDataset.to_uint8); the rest is the internal path."""
    external, _ = _adapter_and_model(dataset_type="external")
    internal, _ = _adapter_and_model(dataset_type="internal")
    rng = np.random.default_rng(0)
    raw = torch.from_numpy((rng.random((1, 7, 20, 11)) * 1000).astype(dtype))
    quantized = torch.from_numpy(LabeledExternalKneeMRIDataset.to_uint8(raw.numpy()))
    assert torch.equal(external.prepare_input(raw), internal.prepare_input(quantized))


@pytest.mark.parametrize("dataset_type", [None, "External", "nifti"])
def test_dinov2_adapter__unknown_dataset_type_raises(dataset_type: str | None) -> None:
    # Anything but "external" would otherwise silently take the internal path.
    with pytest.raises(ValueError, match="dataset_type"):
        DINOv2Adapter(dataset_type=dataset_type, embed_dim=8, image_size=IMAGE_SIZE)  # type: ignore[arg-type]


def test_dinov2_adapter__prepare_input_does_not_mix_slices_at_target_depth() -> None:
    """When the native depth already matches, only the in-plane axes are resized."""
    adapter, _ = _adapter_and_model()
    depth = IMAGE_SIZE[2]
    volume = torch.randint(0, 200, (1, depth, 20, 11), dtype=torch.uint8)
    perturbed = volume.clone()
    perturbed[:, 1] += 50

    changed = (adapter.prepare_input(volume) != adapter.prepare_input(perturbed)).flatten(2)
    assert changed.any(dim=2).squeeze(0).tolist() == [d == 1 for d in range(depth)]


def test_dinov2_adapter__prepare_input_is_deterministic() -> None:
    """Evaluation must not augment: the same volume gives the same input twice."""
    adapter, _ = _adapter_and_model()
    volume = torch.randint(0, 256, (1, 7, 20, 11), dtype=torch.uint8)
    assert torch.equal(adapter.prepare_input(volume), adapter.prepare_input(volume))


def test_dinov2_adapter__forward_features_shapes() -> None:
    adapter, wrapper = _adapter_and_model()
    volume = torch.randint(0, 256, (1, 7, 20, 11), dtype=torch.uint8)
    batch = adapter.collate([adapter.prepare_input(volume) for _ in range(3)])
    assert batch.shape == (3, 1, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1])

    out = adapter.forward_features(wrapper, batch)
    cls = out["cls"]
    patches = out["patches"]
    assert cls is not None and patches is not None
    num_patches = (
        (IMAGE_SIZE[2] // PATCH_SIZE[2])
        * (IMAGE_SIZE[0] // PATCH_SIZE[0])
        * (IMAGE_SIZE[1] // PATCH_SIZE[1])
    )
    assert cls.shape == (3, adapter.embed_dim)
    assert patches.shape == (3, num_patches, adapter.embed_dim)


def test_dinov2_adapter__forward_features_patches_match_wrapper() -> None:
    """The (B, P, C) patches are just the wrapper's (B, C, pD, pH, pW) flattened."""
    adapter, wrapper = _adapter_and_model()
    volume = torch.randint(0, 256, (1, 7, 20, 11), dtype=torch.uint8)
    batch = adapter.collate([adapter.prepare_input(volume) for _ in range(2)])

    out = adapter.forward_features(wrapper, batch)
    cls = out["cls"]
    patches = out["patches"]
    assert cls is not None and patches is not None
    expected = wrapper.forward_features(batch)
    assert torch.allclose(patches, expected["features"].flatten(2).permute(0, 2, 1))
    assert torch.allclose(cls, expected["cls_token"])


def test_dinov2_adapter__is_picklable() -> None:
    """prepare_input runs in DataLoader workers, so the adapter must pickle.

    ClassificationEvaluator wraps it in _AdapterCollate and hands that to the DataLoader
    as collate_fn, which is pickled per worker under spawn/forkserver.
    """
    adapter, _ = _adapter_and_model()
    restored = pickle.loads(pickle.dumps(adapter))
    volume = torch.randint(0, 256, (1, 7, 20, 11), dtype=torch.uint8)
    assert torch.equal(restored.prepare_input(volume), adapter.prepare_input(volume))
