from delayedarray import DelayedArray

from .utils import H5DatasetInfo

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class Hdf5DelayedArray(DelayedArray):
    """Base class for H5 backed arrays.

    May be used to identify a `delayed` object with ``isinstance``.
    """

    @property
    def shape(self) -> tuple:
        """Get shape of the dataset.

        Returns:
            tuple: number of rows by columns.
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
    def order(self) -> str:
        """Get dense matrix format.

        either row-major (C-style) or column-major (Fortran-style) order.

         Returns:
             str: matrix format.
        """
        return self._order

    @property
    def info(self) -> H5DatasetInfo:
        """Get info about the data stored in the H5 File.

        Returns:
            H5DatasetInfo: Namedtuple containing format, dtype and information
            extracted from the H5 file.
        """
        return self._dataset_info

    def __del__(self):
        self._h5file.close()
