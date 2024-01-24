from typing import Optional, Sequence, Tuple, Callable
from delayedarray import extract_dense_array, extract_sparse_array, chunk_shape, DelayedArray, wrap, is_sparse, SparseNdarray
from h5py import File
from numpy import ndarray, dtype, integer, zeros, issubdtype, array
from bisect import bisect_left

__author__ = "LTLA"
__copyright__ = "LTLA"
__license__ = "MIT"


class Hdf5CompressedSparseMatrixSeed:
    """Compressed sparse matrix stored in a HDF5 file, represented as a
    ``DelayedArray`` seed. This assumes that there are three datasets; ``data``
    containing the data values, ``indices`` containing the indices, and
    ``indptr`` containing the pointers to the start of every row/column.
    """

    def __init__(
        self, 
        path: str, 
        group_name: Optional[str], 
        shape: Tuple[int, int],
        by_column: bool,
        dtype: Optional[dtype] = None,
        index_dtype: Optional[dtype] = None,
        data_name: Optional[str] = None,
        indices_name: Optional[str] = None,
        indptr_name: Optional[str] = None,
    ):
        """
        Args:
            path: 
                Path to the HDF5 file.

            group_name:
                Name of the group containing the sparse matrix's contents.
                This can also be None in which case ``data_name``,
                ``indices_name`` and ``indptr_name`` should be specified.

            shape:
                Tuple of length 2 specifying the shape of the matrix.

            by_column:
                Whether this is a compressed sparse column matrix. If False,
                the data is treated as a compressed sparse row matrix.

            dtype:
                NumPy type of the data. Defaults to the HDF5 type on disk;
                otherwise, values are transformed to ``dtype`` during extraction.

            index_dtype:
                NumPy type of the indices. Defaults to the HDF5 type on disk;
                otherwise, values are transformed to ``dtype`` during extraction.

            data_name:
                Name of the dataset containing the data values. Defaults to
                ``group_name`` plus ``/data``.

            indices_name:
                Name of the dataset containing the indices. Defaults to
                ``group_name`` plus ``/indices``.

            indptr_name:
                Name of the dataset containing the pointers. Defaults to
                ``group_name`` plus ``/indptr``.
        """
        self._path = path
        self._group_name = group_name
        self._shape = shape
        self._by_column = by_column

        if data_name is None:
            data_name = group_name + "/data"
        self._data_name = data_name

        if indices_name is None:
            indices_name = group_name + "/indices"
        self._indices_name = indices_name

        if indptr_name is None:
            indptr_name = group_name + "/indptr"
        self._indptr_name = indptr_name

        with File(self._path, "r") as handle:
            self._indptr = handle[self._indptr_name][:]
            if len(self._indptr.shape) != 1 or not issubdtype(self._indptr.dtype, integer):
                raise ValueError("'indptr' dataset should be 1-dimensional and contain integers")
            if by_column:
                if len(self._indptr) != shape[1] + 1:
                    raise ValueError("'indptr' dataset should have length equal to the number of columns + 1")
            else:
                if len(self._indptr) != shape[0] + 1:
                    raise ValueError("'indptr' dataset should have length equal to the number of columns + 1")
            if self._indptr[0] != 0:
                raise ValueError("first entry of 'indptr' dataset should be zero")
            for i in range(1, len(self._indptr)):
                if self._indptr[i] < self._indptr[i-1]:
                    raise ValueError("entries of 'indptr' should be ordered")

            ddset = handle[self._data_name]
            if len(ddset.shape) != 1 or ddset.shape[0] != self._indptr[-1]:
                raise ValueError("'data' dataset should have length equal to the number of non-zero elements")
            self._modify_dtype = (dtype is not None and dtype != ddset.dtype)
            if not self._modify_dtype:
                dtype = ddset.dtype
            self._dtype = dtype

            # Not going to check for consistency of the indices themselves.
            idset = handle[self._indices_name]
            if len(idset.shape) != 1 or idset.shape[0] != self._indptr[-1]:
                raise ValueError("'indices' dataset should have length equal to the number of non-zero elements")
            if not issubdtype(idset.dtype, integer):
                raise ValueError("'indices' dataset should contain integers")
            self._modify_index_dtype = (index_dtype is not None and index_dtype != idset.dtype)
            if not self._modify_index_dtype:
                index_dtype = idset.dtype
            self._index_dtype = index_dtype

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type of this matrix.
        """
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple containing the dimensions of this matrix.
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
    def index_dtype(self) -> dtype:
        """
        Returns:
            NumPy type of the indices in this matrix.
        """
        return self._index_dtype

    @property
    def by_column(self) -> bool:
        """
        Returns:
            Whether the matrix is compressed sparse column.
        """
        return self._by_column

    @property
    def group_name(self) -> Optional[str]:
        """
        Returns:
            Name of the HDF5 group containing the matrix contents, or None if
            the matrix is not contained within a single group.
        """
        return self._group_name

    @property
    def data_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix data values.
        """
        return self._data_name

    @property
    def indices_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix indices.
        """
        return self._indices_name

    @property
    def indptr_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix pointers.
        """
        return self._indptr_name


@is_sparse.register
def is_sparse_Hdf5CompressedSparseMatrixSeed(x: Hdf5CompressedSparseMatrixSeed):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return True


@chunk_shape.register
def chunk_shape_Hdf5CompressedSparseMatrixSeed(x: Hdf5CompressedSparseMatrixSeed):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    if x._by_column:
        return (x._shape[0], 1)
    else:
        return (1, x._shape[1])


def _extract_array(
    x: Hdf5CompressedSparseMatrixSeed, 
    primary_sub: Sequence[int], 
    secondary_sub: Sequence[int], 
    secondary_len: int, 
    f_individual: Callable, 
    f_consecutive: Callable
):
    if len(secondary_sub) == 0:
        return
    secondary_start = secondary_sub[0]
    secondary_end = secondary_sub[-1] + 1
    is_consecutive = (secondary_end - secondary_start == len(secondary_sub))
    search_start = secondary_start > 0
    search_end = secondary_end < secondary_len

    with File(x._path, "r") as handle:
        data = handle[x._data_name]
        indices = handle[x._indices_name]

        for i, p in enumerate(primary_sub):
            start_pos = x._indptr[p]
            end_pos = x._indptr[p + 1]
            curdata = data[start_pos:end_pos]
            curindices = indices[start_pos:end_pos]

            start_idx = 0 
            if search_start:
                start_idx = bisect_left(curindices, secondary_start)
            end_idx = len(curindices)
            if search_end:
                end_idx = bisect_left(curindices, secondary_end, lo=start_idx, hi=end_idx)

            if is_consecutive:
                mod_indices = curindices[start_idx:end_idx]
                if search_start:
                    mod_indices -= secondary_start
                f_consecutive(i, mod_indices, curdata[start_idx:end_idx])
            else:
                p = 0
                for j in range(start_idx, end_idx):
                    curi = curindices[j]
                    while p < len(secondary_sub) and secondary_sub[p] < curi:
                        p += 1
                    if p == len(secondary_sub):
                        break
                    if secondary_sub[p] == curi:
                        f_individual(i, p, curdata[j])
                        p += 1


@extract_dense_array.register
def extract_dense_array_Hdf5CompressedSparseMatrixSeed(x: Hdf5CompressedSparseMatrixSeed, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    if subset is None:
        subset = (range(x.shape[0]), range(x.shape[1]))

    output = zeros((len(subset[0]), len(subset[1])), dtype=x.dtype, order="F")

    if x._by_column: 
        primary_sub = subset[1]
        secondary_sub = subset[0]
        secondary_len = x.shape[0]

        def _individual(c, r, value):
            output[r,c] = value
        def _consecutive(c, rows, values):
            output[rows,c] = values
    else:
        primary_sub = subset[0]
        secondary_sub = subset[1]
        secondary_len = x.shape[1]

        def _individual(r, c, value):
            output[r,c] = value
        def _consecutive(r, cols, values):
            output[r,cols] = values

    _extract_array(
        x=x, 
        primary_sub=primary_sub, 
        secondary_sub=secondary_sub, 
        secondary_len=secondary_len, 
        f_individual=_individual, 
        f_consecutive=_consecutive
    )

    return output


@extract_sparse_array.register
def extract_sparse_array_Hdf5CompressedSparseMatrixSeed(x: Hdf5CompressedSparseMatrixSeed, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    if subset is None:
        subset = (range(x.shape[0]), range(x.shape[1]))

    if x._by_column: 
        primary_sub = subset[1]
        secondary_sub = subset[0]
        primary_len = x.shape[1]
        secondary_len = x.shape[0]
    else:
        primary_sub = subset[0]
        secondary_sub = subset[1]
        primary_len = x.shape[0]
        secondary_len = x.shape[1]

    output = []
    for i in range(len(subset[1])):
        output.append(([], []))

    if x._by_column: 
        def _individual(c, r, value):
            output[c][0].append(r)
            output[c][1].append(value)
        def _consecutive(c, rows, values):
            output[c] = (rows, values)
    else:
        def _individual(r, c, value):
            output[c][0].append(r)
            output[c][1].append(value)
        def _consecutive(r, cols, values):
            for j, c in enumerate(cols):
                output[c][0].append(r)
                output[c][1].append(values[j])

    _extract_array(
        x=x, 
        primary_sub=primary_sub, 
        secondary_sub=secondary_sub, 
        secondary_len=secondary_len, 
        f_individual=_individual, 
        f_consecutive=_consecutive
    )

    all_none = True
    for i, con in enumerate(output):
        if len(con[0]) ==  0:
            output[i] = None
        else:
            output[i] = (array(con[0], dtype=x._index_dtype, copy=False), array(con[1], dtype=x._dtype, copy=False))
            all_none = False
    if all_none:
        output = None

    return SparseNdarray(
        shape=(len(subset[0]), len(subset[1])), 
        contents=output, 
        dtype=x._dtype, 
        index_dtype=x._index_dtype,
        check=False,
    )


class Hdf5CompressedSparseMatrix(DelayedArray):
    """Compressed sparse matrix in a HDF5 file as a ``DelayedArray``. This
    subclass allows developers to implement custom methods for HDF5-backed
    sparse matrices."""

    def __init__(self, path: str, group_name: Optional[str], shape: Tuple[int, int], by_column: bool, **kwargs):
        """
        To construct a ``Hdf5CompressedSparseMatrix`` from an existing
        :py:class:`~Hdf5CompressedSparseMatrixSeed`, use
        :py:meth:`~delayedarray.wrap.wrap` instead.

        Args:
            path: 
                Path to the HDF5 file.

            group_name:
                Name of the dataset containing the array.

            shape:
                Tuple of length 2 specifying the shape of the matrix.

            by_column:
                Whether this is a compressed sparse column matrix. If False,
                the data is treated as a compressed sparse row matrix.

            kwargs:
                Further arguments to pass to the
                :py:class:`~Hdf5CompressedSparseMatrixSeed` constructor.
        """
        if isinstance(path, Hdf5CompressedSparseMatrixSeed):
            seed = path
        else:
            seed = Hdf5CompressedSparseMatrixSeed(path, group_name, shape, by_column, **kwargs)
        super(Hdf5CompressedSparseMatrix, self).__init__(seed)

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the HDF5 file.
        """
        return self.seed.path

    @property
    def index_dtype(self) -> dtype:
        """
        Returns:
            NumPy type of the indices in this matrix.
        """
        return self.seed.index_dtype

    @property
    def by_column(self) -> bool:
        """
        Returns:
            Whether the matrix is compressed sparse column.
        """
        return self.seed.by_column

    @property
    def group_name(self) -> Optional[str]:
        """
        Returns:
            Name of the HDF5 group containing the matrix contents, or None if
            the matrix is not contained within a single group.
        """
        return self.seed.group_name

    @property
    def data_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix data values.
        """
        return self.seed.data_name

    @property
    def indices_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix indices.
        """
        return self.seed.indices_name

    @property
    def indptr_name(self) -> str:
        """
        Returns:
            Name of the HDF5 dataset containing the matrix pointers.
        """
        return self.seed.indptr_name


@wrap.register
def wrap_Hdf5CompressedSparseMatrixSeed(x: Hdf5CompressedSparseMatrixSeed):
    """See :py:meth:`~delayedarray.wrap.wrap`."""
    return Hdf5CompressedSparseMatrix(x, None, None, None)
