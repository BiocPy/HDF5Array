from typing import Optional, Sequence, Tuple, Union
from delayedarray import extract_dense_array, chunk_shape, DelayedArray, wrap, is_masked
from h5py import File
import numpy
from numpy import ndarray, dtype, asfortranarray, ix_

__author__ = "LTLA"
__copyright__ = "LTLA"
__license__ = "MIT"


class Hdf5DenseArraySeed:
    """HDF5-backed dataset as a ``DelayedArray`` dense array seed."""

    def __init__(self, path: str, name: str, dtype: Optional[dtype] = None, native_order: bool = False) -> None:
        """
        Args:
            path: 
                Path to the HDF5 file.

            name:
                Name of the dataset containing the array.

            dtype:
                NumPy type of the data. Defaults to the HDF5 type on disk;
                otherwise, values are transformed to ``dtype`` during extraction.

            native_order: 
                Whether to use HDF5's native order of dimensions. HDF5 orders dimensions
                by slowest to fastest changing. If ``native`` is True, the same ordering
                is used for this array, i.e., this array's shape is the same as that
                reported in the file, equivalent to C storage order.

                If False, this array's shape is reversed compared to that reported in the
                file, equivalent to Fortran storage order. In this case, the first 
                dimension in this array will be the fastest changing one, etc.
        """
        self._path = path
        self._name = name
        self._native_order = native_order

        with File(self._path, "r") as handle:
            dset = handle[name]

            self._modify_dtype = (dtype is not None and dtype != dset.dtype)
            if not self._modify_dtype:
                dtype = dset.dtype
            self._dtype = dtype

            if native_order:
                self._shape = dset.shape
            else:
                self._shape = (*list(reversed(dset.shape)),)

            if dset.chunks is not None:
                if native_order:
                    self._chunks = dset.chunks
                else:
                    self._chunks = (*list(reversed(dset.chunks)),)
            else:
                chunk_sizes = [1] * len(self._shape)
                if native_order:
                    chunk_sizes[-1] = self._shape[-1]
                else:
                    chunk_sizes[0] = self._shape[0]
                self._chunks = (*chunk_sizes,)

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type of this array.
        """
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple containing the dimensions of this array.
        """
        return self._shape

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the HDF5 file.
        """
        return self._path

    @property
    def name(self) -> str:
        """
        Returns:
            Name of the dataset inside the file.
        """
        return self._name


@chunk_shape.register
def chunk_shape_Hdf5DenseArraySeed(x: Hdf5DenseArraySeed):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return x._chunks


@extract_dense_array.register
def extract_dense_array_Hdf5DenseArraySeed(x: Hdf5DenseArraySeed, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    converted = []
    num_lists = 0
    for s in subset:
        if isinstance(s, range): # convert back to slice for HDF5 access efficiency.
            converted.append(slice(s.start, s.stop, s.step))
        else:
            num_lists += 1
            converted.append(s)

    # Currently h5py doesn't support indexing with multiple lists at once.
    # So let's convert all but one of the highest-density entries to slices.
    reextract = None
    if num_lists > 1:
        lowest_density = 1
        chosen = 0
        for i, s in enumerate(converted):
            if not isinstance(s, slice) and len(s):
                lowest = s[1]
                highest = s[-1]
                current_density = (highest - lowest) / len(s)
                if lowest_density > current_density:
                    lowest_density = current_density
                    chosen = i

        reextract = []
        for i, s in enumerate(converted):
            if isinstance(s, slice) or i == chosen:
                reextract.append(range(len(subset[i])))
            else:
                lowest = s[0]
                highest = s[-1]
                converted[i] = slice(lowest, highest + 1)
                reextract.append([j - lowest for j in s])

    # Re-opening the handle as needed, so as to avoid
    # blocking other applications that need this file.
    with File(x._path, "r") as handle:
        dset = handle[x._name]
        if x._native_order:
            out = dset[(*converted,)]
        else:
            converted.reverse()
            out = dset[(*converted,)].T

    if reextract is not None:
        out = out[ix_(*reextract)]

    # Making other transformations for consistency.
    if x._modify_dtype:
        out = out.astype(x._dtype, copy=False)
    if not out.flags.f_contiguous:
        out = asfortranarray(out)
    return out


class Hdf5DenseArray(DelayedArray):
    """HDF5-backed dataset as a ``DelayedArray`` dense array. This subclass
    allows developers to implement custom methods for HDF5-backed arrays."""

    def __init__(self, path: str, name: str, **kwargs):
        """
        To construct a ``Hdf5DenseArray`` from an existing
        :py:class:`~Hdf5DenseArraySeed`, use :py:meth:`~delayedarray.wrap.wrap`
        instead.

        Args:
            path: 
                Path to the HDF5 file.

            name:
                Name of the dataset containing the array.

            kwargs:
                Further arguments to pass to the
                :py:class:`~Hdf5DenseArraySeed` constructor.
        """
        if isinstance(path, Hdf5DenseArraySeed):
            seed = path
        else:
            seed = Hdf5DenseArraySeed(path, name, **kwargs)
        super(Hdf5DenseArray, self).__init__(seed)

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the HDF5 file.
        """
        return self.seed.path

    @property
    def name(self) -> str:
        """
        Returns:
            Name of the dataset inside the file.
        """
        return self.seed.name


@wrap.register
def wrap_Hdf5DenseArraySeed(x: Hdf5DenseArraySeed):
    """See :py:meth:`~delayedarray.wrap.wrap`."""
    return Hdf5DenseArray(x, None)


@is_masked.register
def is_masked_Hdf5DenseArraySeed(x: Hdf5DenseArraySeed) -> bool:
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return False
