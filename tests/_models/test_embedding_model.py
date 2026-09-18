#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from lightly_train._models.embedding_model import EmbeddingModel

from ..helpers import dummy_dinov2_vit_model


class TestEmbeddingModel:
    def test___init___with_embed_dim(self) -> None:
        wrapped_model = dummy_dinov2_vit_model()
        embed_dim = 64
        model = EmbeddingModel(wrapped_model=wrapped_model, embed_dim=embed_dim)
        assert isinstance(model.embed_head, nn.Conv3d)
        assert model.embed_dim == embed_dim

    def test___init___without_embed_dim(self) -> None:
        wrapped_model = dummy_dinov2_vit_model()
        model = EmbeddingModel(wrapped_model=wrapped_model, embed_dim=None)
        assert isinstance(model.embed_head, nn.Identity)
        assert model.embed_dim == wrapped_model.feature_dim()

    def test_forward__with_pooling(self) -> None:
        wrapped_model = dummy_dinov2_vit_model()
        model = EmbeddingModel(wrapped_model=wrapped_model, embed_dim=64)
        x = torch.rand(2, 1, 8, 8, 8)
        output = model(x, pool=True)
        assert output.shape == (2, 64, 1, 1, 1)

    def test_forward__without_pooling(self) -> None:
        wrapped_model = dummy_dinov2_vit_model()
        model = EmbeddingModel(wrapped_model=wrapped_model, embed_dim=64)
        x = torch.rand(2, 1, 8, 8, 8)
        output = model(x, pool=False)
        assert output.shape == (2, 64, 4, 4, 4)

    def test_forward__identity_head(self) -> None:
        wrapped_model = dummy_dinov2_vit_model()
        model = EmbeddingModel(wrapped_model=wrapped_model, embed_dim=None)
        x = torch.rand(2, 1, 8, 8, 8)
        output = model(x, pool=True)
        assert output.shape == (2, 8, 1, 1, 1)
