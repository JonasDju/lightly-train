#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lightly_train._commands import train_helpers
from lightly_train._data import mi_dataset
from lightly_train._data.mi_dataset import MIDataset
from lightly_train._transforms.monai_wrappers import (
    AnisotropyTrackingRandomResizedCrop3D,
)
from lightly_train._transforms.transform import MethodTransform
from lightly_train.types import TransformInput, TransformOutput

from .. import helpers


class RecordingTransform:
    """Records the transform inputs and returns a single view."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, input: TransformInput) -> TransformOutput:
        self.inputs.append(input["image"])
        return [{"image": torch.from_numpy(np.ascontiguousarray(input["image"]))}]


def _dinov2_transform() -> MethodTransform:
    transform_args = train_helpers.get_transform_args(
        method="dinov2", transform_args=dict(helpers.MI_DINOV2_TRANSFORM_ARGS)
    )
    return train_helpers.get_transform(
        method="dinov2", transform_args_resolved=transform_args
    )


def _dinov2_dataset(tmp_path: Path, **kwargs: Any) -> MIDataset:
    data_root, data_meta = helpers.create_mi_dataset(tmp_path, **kwargs)
    return MIDataset(
        data_root=data_root,
        data_meta=data_meta,
        transform=_dinov2_transform(),
        series_depth=8,
    )


class TestMIDataset:
    def test_len(self, tmp_path: Path) -> None:
        data_root, data_meta = helpers.create_mi_dataset(
            tmp_path, n_cases=3, depths=(6, 9)
        )
        dataset = MIDataset(data_root, data_meta, transform=RecordingTransform())
        assert len(dataset) == 6

    @pytest.mark.parametrize(
        "min_series_len, max_series_len, expected_len",
        [(2, None, 4), (7, None, 2), (2, 7, 2)],
    )
    def test_len__series_length_filter(
        self,
        tmp_path: Path,
        min_series_len: int,
        max_series_len: int | None,
        expected_len: int,
    ) -> None:
        data_root, data_meta = helpers.create_mi_dataset(
            tmp_path, n_cases=2, depths=(6, 9)
        )
        dataset = MIDataset(
            data_root,
            data_meta,
            transform=RecordingTransform(),
            min_series_len=min_series_len,
            max_series_len=max_series_len,
        )
        assert len(dataset) == expected_len

    @pytest.mark.parametrize(
        "series_depth, resample_mode, expected_depths, expected_dtype",
        [
            (0, "nearest", {6, 9}, np.uint8),
            (8, "nearest", {8}, np.uint8),
            (8, "interpolate", {8}, np.float32),
        ],
    )
    def test_getitem__transform_input(
        self,
        tmp_path: Path,
        series_depth: int,
        resample_mode: str,
        expected_depths: set[int],
        expected_dtype: type,
    ) -> None:
        data_root, data_meta = helpers.create_mi_dataset(
            tmp_path, n_cases=1, depths=(6, 9), height=20, width=24
        )
        transform = RecordingTransform()
        dataset = MIDataset(
            data_root,
            data_meta,
            transform=transform,
            series_depth=series_depth,
            resample_mode=resample_mode,
        )
        items = [dataset[i] for i in range(len(dataset))]

        # The transform receives (C, H, W, D) arrays.
        assert {x.shape[3] for x in transform.inputs} == expected_depths
        for x in transform.inputs:
            assert x.shape[:3] == (1, 20, 24)
            assert x.dtype == expected_dtype
        for item in items:
            assert item["filename"] == ""
            assert len(item["views"]) == 1

    def test_getitem__permutes_axes(self, tmp_path: Path) -> None:
        data_root, data_meta = helpers.create_mi_dataset(
            tmp_path, n_cases=1, depths=(6,)
        )
        transform = RecordingTransform()
        dataset = MIDataset(data_root, data_meta, transform=transform)
        dataset[0]
        raw = dataset._core[0].numpy()  # (1, D, H, W)
        np.testing.assert_array_equal(transform.inputs[0], raw.transpose(0, 2, 3, 1))

    def test_getitem__dinov2_views(self, tmp_path: Path) -> None:
        dataset = _dinov2_dataset(tmp_path, n_cases=1)
        item = dataset[0]
        views = item["views"]
        assert len(views) == 2 + 2
        for view in views[:2]:
            assert view.shape == (1, 16, 56, 56)
        for view in views[2:]:
            assert view.shape == (1, 8, 28, 28)
        for view in views:
            assert type(view) is torch.Tensor
            assert view.dtype == torch.float32
        # The two global views use differently seeded transforms.
        assert not torch.equal(views[0], views[1])

    def test_pickle(self, tmp_path: Path) -> None:
        dataset = _dinov2_dataset(tmp_path, n_cases=1)
        restored = pickle.loads(pickle.dumps(dataset))
        assert len(restored) == len(dataset)
        assert restored[0]["views"][0].shape == (1, 16, 56, 56)


class TestWorkerSeeding:
    def _first_views_per_worker(self, dataloader: DataLoader) -> list[torch.Tensor]:
        # With batch_size=1 and two workers, consecutive batches come from different
        # workers.
        iterator = iter(dataloader)
        return [next(iterator)["views"][0] for _ in range(2)]

    def test_without_reseeding_workers_repeat_augmentations(
        self, tmp_path: Path
    ) -> None:
        # Sanity check for the test setup: all series are identical, so without
        # reseeding both workers produce exactly the same augmented views.
        dataset = _dinov2_dataset(tmp_path, n_cases=4, depths=(6,), identical=True)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
        view0, view1 = self._first_views_per_worker(dataloader)
        assert torch.equal(view0, view1)

    def test_get_dataloader__reseeds_workers(self, tmp_path: Path) -> None:
        dataset = _dinov2_dataset(tmp_path, n_cases=4, depths=(6,), identical=True)
        dataloader = train_helpers.get_dataloader(
            dataset=dataset,
            batch_size=1,
            num_workers=2,
            series_depth=8,
            loader_args={"shuffle": False},
        )
        assert dataloader.worker_init_fn is mi_dataset.worker_init_fn
        view0, view1 = self._first_views_per_worker(dataloader)
        assert not torch.equal(view0, view1)

    def test_reseed_randomizables__distinct_list_items(self, tmp_path: Path) -> None:
        dataset = _dinov2_dataset(tmp_path, n_cases=1)
        mi_dataset.reseed_randomizables(dataset, rng=np.random.RandomState(0))
        global_0, global_1 = dataset.transform.transforms[:2]  # type: ignore[attr-defined]
        # Found by type, not a fixed index: NormalizeIntensity/ToNumpy now precede
        # the crop in ViewTransform's Compose (see view_transform.py).
        crop_0 = next(
            t
            for t in global_0.transform.transforms
            if isinstance(t, AnisotropyTrackingRandomResizedCrop3D)
        )
        crop_1 = next(
            t
            for t in global_1.transform.transforms
            if isinstance(t, AnisotropyTrackingRandomResizedCrop3D)
        )
        assert crop_0.R is not crop_1.R
        assert crop_0.get_params((1, 64, 64, 16)) != crop_1.get_params((1, 64, 64, 16))

    def test_reseed_randomizables__reproducible(self, tmp_path: Path) -> None:
        dataset = _dinov2_dataset(tmp_path, n_cases=1)
        mi_dataset.reseed_randomizables(dataset, rng=np.random.RandomState(0))
        views_a = dataset[0]["views"]
        mi_dataset.reseed_randomizables(dataset, rng=np.random.RandomState(0))
        views_b = dataset[0]["views"]
        for a, b in zip(views_a, views_b):
            torch.testing.assert_close(a, b)
