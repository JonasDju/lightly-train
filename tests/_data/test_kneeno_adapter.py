#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pickle

import pytest
import torch
from kneeno.evaluation.adapter import EncoderAdapter

from lightly_train._data.kneeno_adapter import DINOv2Adapter

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
    adapter = pickle.loads(pickle.dumps(_adapter()))
    assert adapter.embed_dim == 8


def test_prepare_input__not_implemented() -> None:
    volume = torch.zeros(1, 4, 32, 16, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="stub"):
        _adapter().prepare_input(volume)


def test_forward_features__not_implemented() -> None:
    model = helpers.dummy_dinov2_vit_model()
    with pytest.raises(NotImplementedError, match="stub"):
        _adapter().forward_features(model, torch.zeros(2, 3, 32, 16))
