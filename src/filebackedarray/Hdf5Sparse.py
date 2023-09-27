from typing import Optional, Sequence, Tuple, Union

from delayedarray import extract_dense_array, extract_sparse_array
from h5py import File
from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix

from ._Hdf5Array import Hdf5DelayedArray
from .utils import _check_indices, _slice_h5_sparse, infer_h5_dataset

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class Hdf5SparseArray(Hdf5DelayedArray):
    """HDF5 backed sparse matrix or array store.

    Args:
        path (str): Path to the H5 file.
        group (str): Group inside the file that contains the matrix or array.
    """

    def __init__(self, path_or_file: Union[str, File], group: str) -> None:
        """Initialize a H5 Backed array.

        Args:
            path_or_file (str): Path to the H5 file.
            group (str): Group inside the file that contains the matrix or array.
        """
        self._h5file = path_or_file
        if isinstance(path_or_file, str):
            self._h5file = File(path_or_file, mode="r")

        self._seed = self._h5file[group]

        # TODO: If this gets too complicated, might have to add a
        # parameter that specifies the matrix format instead of inferring it
        # from the file.
        self._dataset_info = infer_h5_dataset(self._dataset)

        if self._dataset_info.format not in ["csr_matrix", "csc_matrix"]:
            raise ValueError("File does not contain a sparse matrix")


@extract_sparse_array.register(Hdf5SparseArray)
def _extract_sparse_h5array(
    x: Hdf5SparseArray, subset: Optional[Tuple[Sequence[int]]] = None
) -> Union[csr_matrix, csc_matrix]:
    if len(subset) == 0:
        raise ValueError("Arguments must contain one slice")

    rowIndices = _check_indices(subset[0])
    colIndices = None

    if len(subset) > 1:
        if subset[1] is not None:
            colIndices = _check_indices(subset[1])
    elif len(subset) > 2:
        raise ValueError("contains too many slices")

    if x.order == "csr_matrix":
        mat = _slice_h5_sparse(x.seed, x._dataset_info, rowIndices)
        # now slice columns
        if colIndices is not None:
            mat = mat[:, colIndices]
        return mat
    elif x.order == "csc_matrix":
        if colIndices is None:
            colIndices = slice(0)
        mat = _slice_h5_sparse(x.seed, x._dataset_info, colIndices)
        # now slice columns
        mat = mat[rowIndices, :]
        return mat
    else:
        raise Exception("unknown matrix type in H5.")


@extract_dense_array.register(Hdf5SparseArray)
def _extract_sparse_as_dense_h5array(
    x: Hdf5SparseArray, subset: Optional[Tuple[Sequence[int]]] = None
) -> ndarray:
    mat = _extract_sparse_as_dense_h5array(x, subset)
    return mat.todense()
