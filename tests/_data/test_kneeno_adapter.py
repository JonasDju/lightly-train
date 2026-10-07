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
    return DINOv2Adapter(dataset_type=dataset_type, embed_dim=8, image_size=(32, 16))


def test_is_encoder_adapter() -> None:
    assert isinstance(_adapter(), EncoderAdapter)


def test_has_cls_token() -> None:
    # DINOv2 has a cls token, so KneeNo's "linear" task is available.
    assert DINOv2Adapter.has_cls_token is True


def test_init() -> None:
    adapter = DINOv2Adapter(
        dataset_type="external",
        embed_dim=8,
        image_size=[32, 16],
        normalize=((0.5,), (0.25,)),
    )
    assert adapter.embed_dim == 8
    assert adapter.dataset_type == "external"
    assert adapter.image_size == (32, 16)
    assert adapter.mean == (0.5,)
    assert adapter.std == (0.25,)


def test_init__invalid_dataset_type() -> None:
    with pytest.raises(ValueError, match="dataset_type"):
        _adapter(dataset_type="labeled")


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
