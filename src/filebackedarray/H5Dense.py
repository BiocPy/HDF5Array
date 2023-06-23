from typing import Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from .backedarray import BackedArray
from .utils.h5utils import _check_indices, infer_dataset

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

ORDER_OPTS = ["C", "F"]


class H5DenseArray(BackedArray):
    """H5 backed dense array or matrices.

    Currently tested for 2-dimensional matrices, if n > 2, use with caution.

    Args:
        path (str): Path to the H5 file.
        group (str): Group inside the file that contains the matrix or array.
        order (Literal["C", "F"]): dense matrix representation, ‘C’, ‘F’,
            row-major (C-style) or column-major (Fortran-style) order.
    """

    def __init__(self, path: str, group: str, order: Literal["C", "F"] = "C") -> None:
        """Initialize a H5 Backed array."""
        super().__init__()

        self._h5file = h5py.File(path, mode="r")
        self._dataset = self._h5file[group]
        self._dataset_info = infer_dataset(self._dataset)

        if order not in ORDER_OPTS:
            raise ValueError(
                "order must be `C` (c-style, row-major) or "
                "`F` (fortran-style, column-major)"
            )

        self._order = order

        if self._dataset_info.format != "dense":
            raise ValueError("File does not contain a dense matrix")

    @property
    def order(self) -> Literal["C", "F"]:
        """Get order of the dense matrix.

        row-major (C-style) or column-major (Fortran-style) order.

        Returns:
            Literal["C", "F"]: either ‘C’, ‘F’.
        """
        return self._order

    @property
    def mat_format(self) -> Literal["C", "F"]:
        """Get the dense matrix format.

        either row-major (C-style) or column-major (Fortran-style) order.

         Returns:
            Literal["C", "F"]: either ‘C’, ‘F’.
        """
        return self._order

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the dataset.

        Returns:
            Tuple[int, int]: number of rows by columns.
        """
        _shape = None

        if self.order == "C":
            _shape = self._dataset_info.shape
        else:
            _shape = self._dataset_info.shape[::-1]

        # technically if the orientation is F and transposed is True,
        # this is C but watever
        if self.transposed:
            return _shape[::-1]

        return _shape

    @property
    def dtype(self) -> str:
        """Get type of values stored in the dataset.

        Returns:
            str: type of dataset, e.g. int8, float etc.
        """
        return self._dataset_info.dtype

    def __getitem__(
        self,
        args: Tuple[Union[slice, Sequence[int]], Optional[Union[slice, Sequence[int]]]],
    ) -> np.ndarray:
        """Get the slice from the H5 file.

        Args:
            args (Tuple[Union[slice, Sequence[int]], ...]):
                slices along each dimension.

        Raises:
            ValueError: if enough slices are not provided
            ValueError: provided too many slices

        Returns:
            np.ndarray: numpy ndarray of the slice.
        """
        if len(args) == 0:
            raise ValueError("Arguments must contain atleast one slice")

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


class H5BackedDenseData(H5DenseArray):
    """Alias to :class:`H5DenseArray` for backwards compatibility."""

    pass
