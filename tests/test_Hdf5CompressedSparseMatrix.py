import numpy
import h5py
from filebackedarray import Hdf5CompressedSparseMatrix
import delayedarray
import tempfile
import scipy.sparse

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _mockup(mat):
    _, path = tempfile.mkstemp(suffix=".h5")
    name = "whee"

    with h5py.File(path, "w") as handle:
        handle.create_dataset(name + "/data", data=mat.data, compression="gzip")
        handle.create_dataset(name + "/indices", data=mat.indices, compression="gzip")
        handle.create_dataset(name + "/indptr", data=mat.indptr, compression="gzip")

    return path, name


def test_Hdf5CompressedSparseMatrix_column():
    shape = (100, 80)
    y = scipy.sparse.random(*shape, 0.1).tocsc()
    path, group = _mockup(y)
    arr = Hdf5CompressedSparseMatrix(path, group, shape=shape, by_column=True)

    assert arr.shape == shape
    assert arr.dtype == y.dtype
    assert delayedarray.chunk_shape(arr) == (100, 1)
    assert (delayedarray.extract_dense_array(arr) == y.toarray()).all()

    # Check that consecutive slicing works as expected.
    slices = (slice(30, 90), slice(20, 60))
    ranges = [range(*s.indices(shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == y[slices].toarray()).all()

    # Check that non-consecutive slicing works as expected.
    slices = (slice(3, 90, 3), slice(4, 70, 5))
    ranges = [range(*s.indices(shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == y[slices].toarray()).all()


def test_Hdf5CompressedSparseMatrix_row():
    shape = (100, 200)
    y = scipy.sparse.random(*shape, 0.1).tocsr()
    path, group = _mockup(y)
    arr = Hdf5CompressedSparseMatrix(path, group, shape=shape, by_column=False)

    assert arr.shape == shape 
    assert arr.dtype == y.dtype
    assert delayedarray.chunk_shape(arr) == (1, 200)
    assert (delayedarray.extract_dense_array(arr) == y.toarray()).all()

    # Check that consecutive slicing works as expected.
    slices = (slice(10, 80), slice(50, 150))
    ranges = [range(*s.indices(shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == y[slices]).all()

    # Check that non-consecutive slicing works as expected.
    slices = (slice(10, 80, 2), slice(50, 150, 3))
    ranges = [range(*s.indices(shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == y[slices]).all()
