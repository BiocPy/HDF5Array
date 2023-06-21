# Tutorial

This package provides classes to support file backed arrays or matrices stored in H5 files. We'll soon provide similar implementations for tiledb, zarr and other matrix storage formats.

```python
from filebackedarray import H5BackedAssay

matrix = H5BackedAssay("tests/data/tenx.sub.h5", group="matrix")

# get the dimensions of the matrix
print("matrix shape: ", matrix.shape)

# slice the matrix
matrix_slice = matrix[0:100, 1:101]
```