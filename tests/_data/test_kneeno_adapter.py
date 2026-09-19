#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pickle

import torch
from kneeno.evaluation.adapter import EncoderAdapter

from lightly_train._data.kneeno_adapter import DINOv2Adapter
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper

from .. import helpers

# Non-cubic on purpose: H != W != D is what catches an (H, W, D) <-> (D, H, W) swap,
# which is invisible on cubic fixtures. See CLAUDE.md's axis-order gotcha.
IMAGE_SIZE = (16, 8, 4)  # (H, W, D)
PATCH_SIZE = (2, 2, 2)


def _adapter_and_model() -> tuple[DINOv2Adapter, DINOv2ViTModelWrapper]:
    wrapper = helpers.dummy_dinov2_vit_model(patch_size=PATCH_SIZE, img_size=IMAGE_SIZE)
    adapter = DINOv2Adapter(embed_dim=wrapper.feature_dim(), image_size=IMAGE_SIZE)
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
        embed_dim=8, image_size=IMAGE_SIZE, normalize=((0.25,), (0.5,))
    )
    # Already the target size, so no interpolation runs and values map exactly.
    volume = torch.full((1, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1]), 255.0)
    out = adapter.prepare_input(volume)
    expected = (255.0 - 0.25 * 255.0) / (0.5 * 255.0)
    assert torch.allclose(out, torch.full_like(out, expected))


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
