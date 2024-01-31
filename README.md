<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/hdf5array.svg?branch=main)](https://cirrus-ci.com/github/<USER>/hdf5array)
[![ReadTheDocs](https://readthedocs.org/projects/hdf5array/badge/?version=latest)](https://hdf5array.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/hdf5array/main.svg)](https://coveralls.io/r/<USER>/hdf5array)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/hdf5array.svg)](https://anaconda.org/conda-forge/hdf5array)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/hdf5array)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/hdf5array.svg)](https://pypi.org/project/hdf5array/)
[![Monthly Downloads](https://pepy.tech/badge/hdf5array/month)](https://pepy.tech/project/hdf5array)
![Unit tests](https://github.com/BiocPy/hdf5array/actions/workflows/pypi-test.yml/badge.svg)


# hdf5array

## Introduction

This is the Python equivalent of Bioconductor's [**HDF5Array**](https://bioconductor.org/packages/HDF5Array) package,
providing a representation of HDF5-backed arrays within the [**delayedarray**](https://github.com/BiocPy/delayedarray) framework.
The idea is to allow users to store, manipulate and operate on large datasets without loading them into memory,
in a manner that is trivially compatible with other data structures in the [**BiocPy**](https::/github.com/BiocPy) ecosystem.

## Installation

This package can be installed from [PyPI](https://pypi.org/project/hdf5array/) with the usual commands:

```shell
pip install hdf5array
```

## Quick start

Let's mock up a dense array:

```python
import numpy
data = numpy.random.rand(40, 50, 100)

import h5py
with h5py.File("whee.h5", "w") as handle:
    handle.create_dataset("yay", data=data)
```

We can now represent it as a `Hdf5DenseArray`:

```python
import hdf5array
arr = hdf5array.Hdf5DenseArray("whee.h5", "yay", native_order=True)
## <40 x 50 x 100> Hdf5DenseArray object of type 'float64'
## [[[0.63008796, 0.34849183, 0.75621679, ..., 0.07343495, 0.63095765,
##    0.625732  ],
##   [0.68123095, 0.91403054, 0.74737122, ..., 0.17344344, 0.82254404,
##    0.58158815],
##   [0.83287116, 0.40738123, 0.89887551, ..., 0.34936481, 0.76600276,
##    0.91991967],
##   ...,
```

This is just a subclass of a `DelayedArray` and can be used anywhere in the BiocPy framework.
Parts of the NumPy API are also supported - for example, we could apply a variety of delayed operations:

```python
scaling = numpy.random.rand(100)
transformed = numpy.log1p(arr / scaling)
## <40 x 50 x 100> DelayedArray object of type 'float64'
## [[[0.58803887, 0.3458478 , 0.82700531, ..., 0.08224734, 0.65678967,
##    0.56893312],
##   [0.62348907, 0.7341526 , 0.82040225, ..., 0.18437718, 0.7932422 ,
##    0.53784637],
##   [0.72176703, 0.39407341, 0.92788307, ..., 0.34205035, 0.75487196,
##    0.75456938],
##   ...,
```

Check out the [documentation](https://biocpy.github.io/hdf5array/) for more details.

## Handling sparse matrices

We support a variety of compressed sparse formats where the non-zero elements are held inside three separate datasets -
usually `data`, `indices` and `indptr`, based on the [10X Genomics sparse HDF5 format](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices).
To demonstrate, let's mock up some sparse data using **scipy**:

```python
import scipy.sparse
mock = scipy.sparse.random(1000, 200, 0.1).tocsc()

with h5py.File("sparse_whee.h5", "w") as handle:
    handle.create_dataset("sparse_blah/data", data=mock.data, compression="gzip")
    handle.create_dataset("sparse_blah/indices", data=mock.indices, compression="gzip")
    handle.create_dataset("sparse_blah/indptr", data=mock.indptr, compression="gzip")
```

We can then create a sparse HDF5-backed matrix.
Note that there is some variation in this HDF5 compressed sparse format, notably where the dimensions are stored and whether it is column/row-major.
The constructor will not do any auto-detection so we need to provide this information explicitly:

```python
import hdf5array
arr = hdf5array.Hdf5CompressedSparseMatrix(
    "sparse_whee.h5", 
    "sparse_blah", 
    shape=(100, 200), 
    by_column=True
)
## <100 x 200> sparse Hdf5CompressedSparseMatrix object of type 'float64'
## [[0.        , 0.        , 0.26563417, ..., 0.        , 0.        ,
##   0.        ],
##  [0.        , 0.        , 0.        , ..., 0.23896924, 0.        ,
##   0.        ],
##  [0.        , 0.        , 0.        , ..., 0.42236848, 0.3585153 ,
##   0.        ],
##  ...,
##  [0.        , 0.        , 0.3363087 , ..., 0.        , 0.        ,
##   0.        ],
##  [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
##   0.        ],
##  [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
##   0.        ]]
```
