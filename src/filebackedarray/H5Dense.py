from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from .utils import _check_indices, infer_h5_dataset

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class H5BackedDenseData:
    """H5 backed dense matrix or array store.

    Args:
        path (str): Path to the H5 file.
        group (str): Group inside the file that contains the matrix or array.
        order (str): dense matrix representation, ‘C’, ‘F’,
            row-major (C-style) or column-major (Fortran-style) order.
    """

    def __init__(self, path: str, group: str, order: str = "C") -> None:
        """Initialize a H5 Backed array.

        Args:
            path (str): Path to the H5 file.
            group (str): Group inside the file that contains the matrix or array.
            order (str): dense matrix representation, ‘C’, ‘F’,
                row-major (C-style) or column-major (Fortran-style) order.
        """
        self._h5file = h5py.File(path, mode="r")
        self._dataset = self._h5file[group]
        self._dataset_info = infer_h5_dataset(self._dataset)

        if order not in ("C", "F"):
            raise ValueError(
                "order must be C (c-style, row-major) or F (fortran-style, column-major)"
            )

        self._order = order

        if self._dataset_info.format != "dense":
            raise ValueError("File does not contain a dense matrix")

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the dataset.

        Returns:
            Tuple[int, int]: number of rows by columns.
        """
        if self._order == "C":
            return self._dataset_info.shape
        else:
            return self._dataset_info.shape[::-1]

    @property
    def dtype(self) -> str:
        """Get type of values stored in the dataset.

        Returns:
            str: type of dataset, e.g. int8, float etc.
        """
        return self._dataset_info.dtype

    @property
    def mat_format(self) -> str:
        """Get dense matrix format.

        either row-major (C-style) or column-major (Fortran-style) order.

         Returns:
             str: matrix format.
        """
        return self._order

    def __getitem__(
        self,
        args: Tuple[Union[slice, Sequence[int]], Optional[Union[slice, Sequence[int]]]],
    ) -> np.ndarray:
        if len(args) == 0:
            raise ValueError("Arguments must contain one slice")

        rowIndices = _check_indices(args[0])
        colIndices = None

        if len(args) > 1:
            if args[1] is not None:
                colIndices = _check_indices(args[1])
        elif len(args) > 2:
            raise ValueError("contains too many slices")

        if colIndices is None:
            colIndices = slice(0)

        if self.mat_format == "C":
            return self._dataset[rowIndices, colIndices]
        else:
            return self._dataset[colIndices, rowIndices]

    # TODO: switch to weak refs at some point
    def __del__(self):
        self._h5file.close()
