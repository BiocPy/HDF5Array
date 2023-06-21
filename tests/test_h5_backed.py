import scipy.sparse as sp
from filebackedarray import H5Backed

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_h5_backed():
    assay = H5Backed("tests/data/tenx.sub.h5", "matrix")

    assert assay is not None
    assert isinstance(assay, H5Backed)
    assert assay.shape == (1000, 3005)
    assert assay.mat_format == "csc_matrix"
    assert assay.dtype is not None


def test_h5_backed_slice():
    assay = H5Backed("tests/data/tenx.sub.h5", "matrix")

    assert assay is not None
    assert isinstance(assay, H5Backed)
    assert assay.shape == (1000, 3005)

    asy_slice = assay[0:100, 1:101]
    assert isinstance(asy_slice, sp.spmatrix)
    assert asy_slice.shape == (100, 100)
