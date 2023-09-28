import numpy
import h5py
from filebackedarray import Hdf5DenseArray
import delayedarray
import tempfile

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _mockup(data, chunks, compression):
    _, path = tempfile.mkstemp(suffix=".h5")
    name = "whee"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=data, chunks=chunks, compression=compression)
    return path, name


def test_Hdf5DenseArray_native():
    test_shape = (100, 200)
    y = numpy.random.rand(*test_shape)
    chunk_sizes = (10, 20)
    path, name = _mockup(y, chunk_sizes, 'gzip')
    arr = Hdf5DenseArray(path, name, native_order=True)

    assert arr.shape == y.shape
    assert arr.dtype == y.dtype
    assert delayedarray.chunk_shape(arr) == chunk_sizes
    assert (delayedarray.extract_dense_array(arr) == y).all()

    # Check that the slicing works as expected.
    slices = (slice(3, 90, 3), slice(4, 160, 5))
    ref = y[slices]
    ranges = [range(*s.indices(test_shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == ref).all()

    # Check that it works with explicit lists.
    ranges2 = [list(r) for r in ranges]
    assert (delayedarray.extract_dense_array(arr, (*ranges2,)) == ref).all()

    # Check that it works with a mix of both.
    mixed = (ranges[0], slices[1])
    assert (delayedarray.extract_dense_array(arr, (*mixed,)) == ref).all()


def test_Hdf5DenseArray_non_native():
    test_shape = (100, 200)
    y = numpy.random.rand(*test_shape)
    chunk_sizes = (10, 20)
    path, name = _mockup(y, chunk_sizes, 'gzip')
    arr = Hdf5DenseArray(path, name, native_order=False)

    actual_shape = (*list(reversed(test_shape)),)
    actual_chunk_sizes = (*list(reversed(chunk_sizes)),)
    assert arr.shape == actual_shape
    assert arr.dtype == y.dtype
    assert delayedarray.chunk_shape(arr) == actual_chunk_sizes
    assert (delayedarray.extract_dense_array(arr) == y.T).all()

    # Check that the slicing works as expected.
    slices = (slice(100, 180), slice(50, 100))
    ref = y.T[slices]
    ranges = [range(*s.indices(actual_shape[i])) for i, s in enumerate(slices)]
    assert (delayedarray.extract_dense_array(arr, (*ranges,)) == ref).all()

    # Check that it works with explicit lists.
    ranges2 = [list(r) for r in ranges]
    assert (delayedarray.extract_dense_array(arr, (*ranges2,)) == ref).all()

    # Check that it works with a mix of both.
    mixed = (slices[0], ranges[1])
    assert (delayedarray.extract_dense_array(arr, (*mixed,)) == ref).all()


def test_Hdf5DenseArray_new_type():
    test_shape = (100, 200)
    y = numpy.random.rand(*test_shape) * 10
    chunk_sizes = (10, 20)
    path, name = _mockup(y, chunk_sizes, 'gzip')
    arr = Hdf5DenseArray(path, name, dtype=numpy.dtype("int32"), native_order=True)

    assert arr.shape == test_shape
    assert arr.dtype == numpy.dtype("int32")
    assert delayedarray.chunk_shape(arr) == chunk_sizes
    assert (delayedarray.extract_dense_array(arr) == y.astype(numpy.int32)).all()


def test_Hdf5DenseArray_properties():
    test_shape = (100, 200)
    y = numpy.random.rand(*test_shape) * 10
    chunk_sizes = (10, 20)
    path, name = _mockup(y, chunk_sizes, 'gzip')
    arr = Hdf5DenseArray(path, name, dtype=numpy.dtype("int32"), native_order=True)

    assert arr.path == path
    assert arr.name == name

    rewrap = delayedarray.wrap(arr.seed)
    assert isinstance(rewrap, Hdf5DenseArray)
