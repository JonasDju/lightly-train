from __future__ import annotations

from typing import Any

import numpy as np
import torch
from kneeno.dataset import UnlabeledKneeMRIDataset
from monai.transforms import Randomizable
from monai.utils import MAX_SEED
from torch.utils.data import Dataset

from lightly_train.types import DatasetItem, PathLike, Transform, TransformInput


class MIDataset(Dataset[DatasetItem]):

    def __init__(
            self,
            data_root: PathLike,
            data_meta: PathLike,
            transform: Transform,
            series_depth: int = 0,
            resample_mode: str = "nearest",
            min_series_len: int = 2,
            max_series_len: int | None = None
    ):
        self._core = UnlabeledKneeMRIDataset(
            data_root, data_meta, series_depth, resample_mode, min_series_len, max_series_len
        )
        self.transform = transform
        # MONAI's Randomizable transforms share a class-level RandomState until they are
        # seeded. Seed each one individually from torch's RNG so that the augmentations
        # are reproducible with pytorch_lightning.seed_everything.
        reseed_randomizables(
            self, rng=np.random.RandomState(int(torch.randint(0, MAX_SEED, ()).item()))
        )

    def __len__(self):
        return len(self._core)

    def __getitem__(self, index) -> DatasetItem:
        volume = self._core[index]                  # (1, D, H, W)

        # The monai transforms expect C, H, W, D
        volume = volume.permute(0, 2, 3, 1).numpy() # (1, H, W, D)

        input: TransformInput = {"image": volume}
        transformed = self.transform(input)         # (1, D, H, W)

        dataset_item: DatasetItem = {
            "filename": "",                         # Todo: see if this breaks something
            "views": [view["image"] for view in transformed]
        }
        return dataset_item


def reseed_randomizables(obj: Any, rng: np.random.RandomState) -> None:
    """Give every MONAI ``Randomizable`` reachable from ``obj`` its own random state.

    Unlike ``monai.data.utils.set_rnd``, items inside lists get distinct seeds. This
    matters for DINO-style transforms whose views are stored in a list: with equal
    seeds the two global views would get identical crops.
    """
    _reseed_randomizables(obj, rng=rng, seen=set())


def _reseed_randomizables(obj: Any, rng: np.random.RandomState, seen: set[int]) -> None:
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, Randomizable):
        # MONAI's Compose propagates distinct seeds to its children.
        obj.set_random_state(seed=int(rng.randint(0, MAX_SEED)))
        return
    if isinstance(obj, (list, tuple)):
        children: Any = obj
    elif isinstance(obj, dict):
        children = obj.values()
    elif hasattr(obj, "__dict__") and not isinstance(obj, (torch.Tensor, np.ndarray)):
        children = vars(obj).values()
    else:
        return
    for child in children:
        _reseed_randomizables(child, rng=rng, seen=seen)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that reseeds the dataset's random transforms.

    Without this, every worker (and, with non-persistent workers, every epoch) starts
    from the same copied random state and produces identical augmentations. Torch and
    Lightning only reseed the global ``random``/``numpy``/``torch`` generators, not the
    per-transform ``RandomState`` held by MONAI transforms.
    """
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None
    reseed_randomizables(
        worker_info.dataset, rng=np.random.RandomState(worker_info.seed % MAX_SEED)
    )
