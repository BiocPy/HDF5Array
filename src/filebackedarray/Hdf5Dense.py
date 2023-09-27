from typing import Optional, Sequence, Tuple, Union

from delayedarray import extract_dense_array
from h5py import File
from numpy import ndarray

from ._Hdf5Array import Hdf5DelayedArray
from .utils import _check_indices, infer_h5_dataset

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class Hdf5DenseArray(Hdf5DelayedArray):
    """HDF5 backed dense matrix or array store.

    Args:
        path (str): Path to the H5 file.
        group (str): Group inside the file that contains the matrix or array.
        order (str): dense matrix representation, ‘C’, ‘F’,
            row-major (C-style) or column-major (Fortran-style) order.
    """

    def __init__(
        self, path_or_file: Union[str, File], group: str, order: str = "C"
    ) -> None:
        """Initialize a H5 Backed array.

        Args:
            path_or_file (Union[str, File]): Path to the H5 file.
            group (str): Group inside the file that contains the matrix or array.
            order (str): dense matrix representation, ‘C’, ‘F’,
                row-major (C-style) or column-major (Fortran-style) order.
        """
        self._h5file = path_or_file
        if isinstance(path_or_file, str):
            self._h5file = File(path_or_file, mode="r")

        self._seed = self._h5file[group]
        self._dataset_info = infer_h5_dataset(self._seed)

        if order not in ("C", "F"):
            raise ValueError(
                "order must be C (c-style, row-major) or F (fortran-style, column-major)"
            )

        self._order = order

        if self._dataset_info.format != "dense":
            raise ValueError("File does not contain a dense matrix")


@extract_dense_array.register(Hdf5DenseArray)
def _extract_hdf5_dense(
    x: Hdf5DenseArray, subset: Optional[Tuple[Sequence[int]]] = None
) -> ndarray:
    if len(subset) == 0:
        raise ValueError("Arguments must contain atleast one slice!")

    rowIndices = _check_indices(subset[0])
    colIndices = None

    if len(subset) > 1:
        if subset[1] is not None:
            colIndices = _check_indices(subset[1])
    elif len(subset) > 2:
        raise ValueError("contains too many slices")

    if colIndices is None:
        colIndices = slice(0)

    mat = x.seed[rowIndices, colIndices]

    if x.order != "F":
        mat = mat.T

    return mat
