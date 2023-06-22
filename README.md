<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/FileBackedArray.svg?branch=main)](https://cirrus-ci.com/github/<USER>/FileBackedArray)
[![ReadTheDocs](https://readthedocs.org/projects/FileBackedArray/badge/?version=latest)](https://FileBackedArray.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/FileBackedArray/main.svg)](https://coveralls.io/r/<USER>/FileBackedArray)
[![PyPI-Server](https://img.shields.io/pypi/v/FileBackedArray.svg)](https://pypi.org/project/FileBackedArray/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/FileBackedArray.svg)](https://anaconda.org/conda-forge/FileBackedArray)
[![Monthly Downloads](https://pepy.tech/badge/FileBackedArray/month)](https://pepy.tech/project/FileBackedArray)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/FileBackedArray)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# FileBackedArray

Python package to support file backed arrays and matrices stored in H5 file. Will soon support tiledb and zarr based formats.

## Install

Package is published to [PyPI](https://pypi.org/project/filebackedarray/)

```shell
pip install filebackedarray
```

## Usage

```python
from filebackedarray import H5BackedSparseData

matrix = H5BackedSparseData("tests/data/tenx.sub.h5", group="matrix")

# get the dimensions of the matrix
print("matrix shape: ", matrix.shape)

# slice the matrix
matrix_slice = matrix[0:100, 1:101]
```

Checkout the [documentation](https://biocpy.github.io/FileBackedArray/) for more info.



<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
