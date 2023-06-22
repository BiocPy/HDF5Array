import numpy as np
from filebackedarray import H5BackedDenseData

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_h5_dense_backed_C():
    assay = H5BackedDenseData("tests/data/dense.h5", "dense_C")

    assert assay is not None
    assert isinstance(assay, H5BackedDenseData)
    assert assay.shape == (100, 100)
    assert assay.mat_format == "C"
    assert assay.dtype is not None

    asy_slice = assay[0:2, 1:4]
    assert isinstance(asy_slice, np.ndarray)
    assert asy_slice.shape == (2, 3)


def test_h5_dense_backed_F():
    assay = H5BackedDenseData("tests/data/dense_F.h5", "dense_F", order="F")

    assert assay is not None
    assert isinstance(assay, H5BackedDenseData)
    assert assay.shape == (3, 2)
    assert assay.mat_format == "F"

    asy_slice = assay[0:1, 0:2]
    assert isinstance(asy_slice, np.ndarray)
    assert asy_slice.shape == (2, 1)
