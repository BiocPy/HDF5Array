from typing import Optional, Sequence, Tuple, Union

import h5py

from .utils import _check_indices, _slice_h5_sparse, infer_h5_dataset


class H5BackedData:
    """H5 backed matrix or array store.

    Args:
        path (str): Path to the H5 file.
        group (str): Group inside the file that contains the matrix or array.
    """

    def __init__(self, path: str, group: str) -> None:
        """Initialize a H5 Backed array.

        Args:
            path (str): Path to the H5 file.
            group (str): Group inside the file that contains the matrix or array.
        """
        self._h5file = h5py.File(path, mode="r")
        self._dataset = self._h5file[group]

        self._dataset_info = infer_h5_dataset(self._dataset)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the dataset.

        Returns:
            Tuple[int, int]: number of rows by columns.
        """
        return self._dataset_info.shape

    @property
    def dtype(self) -> str:
        """Get type of values stored in the dataset.

        Returns:
            str: type of dataset, e.g. int8, float etc.
        """
        return self._dataset_info.dtype

    @property
    def mat_format(self) -> str:
        """Get matrix format of the dataset.

        either `csr_matrix`, `csc_matrix` or `dense`.

         Returns:
             str: matrix format.
        """
        return self._dataset_info.format

    def __getitem__(
        self,
        args: Tuple[Union[slice, Sequence[int]], Optional[Union[slice, Sequence[int]]]],
    ):
        if len(args) == 0:
            raise ValueError("Arguments must contain one slice")

        rowIndices = _check_indices(args[0])
        colIndices = None

        if len(args) > 1:
            if args[1] is not None:
                colIndices = _check_indices(args[1])
        elif len(args) > 2:
            raise ValueError("contains too many slices")

        if self.mat_format == "csr_matrix":
            mat = _slice_h5_sparse(self._dataset, self._dataset_info, rowIndices)
            # now slice columns
            if colIndices is not None:
                mat = mat[:, colIndices]
            return mat
        elif self.mat_format == "csc_matrix":
            if colIndices is None:
                colIndices = slice(0)
            mat = _slice_h5_sparse(self._dataset, self._dataset_info, colIndices)
            # now slice columns
            mat = mat[rowIndices, :]
            return mat
        elif self.mat_format == "dense":
            if colIndices is None:
                colIndices = slice(0)
            return self._dataset[rowIndices, colIndices]
        else:
            raise Exception("unknown matrix type in H5.")

    # TODO: switch to weak refs at some point
    def __del__(self):
        self._h5file.close()
