from torch.utils.data import Dataset
from kneeno.dataset import UnlabeledKneeMRIDataset
from torch.utils.data.dataset import _T_co

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

    def __len__(self):
        return len(self._core)

    def __getitem__(self, index) -> DatasetItem:
        volume = self._core[index] # (1, D, H, W)

        input: TransformInput = {"image": volume}
        transformed = self.transform(input)         # Todo: check if Transform expects channel first. The output must
                                                    #       be channel first.

        dataset_item: DatasetItem = {
            "filename": "",                         # Todo: see if this breaks something
            "views": [view["image"] for view in transformed]
        }
        return dataset_item

