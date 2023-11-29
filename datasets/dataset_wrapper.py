import os
from typing import List, Union

__all__ = ["BaseDataset"]


class DatasetWrapper(object):
    """
    Base dataset to parse dataset files.

    Args:
        - batch_data:
    Attributes:
        data_list (List(Tuple)): source data items (e.g., containing image path and raw annotation)
    """

    def __init__(self, batch_data):
        self._index = 0
        self.data_list = [batch_data]
        # WARNING: shallow copy. Do deep copy if necessary.
        # _data = self.data_list[0].copy()
        self.output_columns = ["data"]

    def __getitem__(self, index):
        # WARNING: shallow copy. Do deep copy if necessary.
        data = self.data_list[index].copy()
        return data

    def set_output_columns(self, column_names: List[str]):
        self.output_columns = column_names

    def get_output_columns(self) -> List[str]:
        """
        get the column names for the output tuple of __getitem__, required for data mapping in the next step
        """
        # raise NotImplementedError
        return self.output_columns

    def __next__(self):
        if self._index >= len(self.data_list):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return len(self.data_list)

    def _load_image_bytes(self, img_path):
        """load image bytes (prepared for decoding)"""
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        return image_bytes
