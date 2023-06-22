# Tutorial

This package provides classes to support file backed arrays or matrices stored in H5 files. We'll soon provide similar implementations for tiledb, zarr and other matrix storage formats.

## Sparse matrices

`H5BackedSparseData` tries to infer the sparse matrix format, either `csr_matrix` or `csc_matrix` from the h5.

```python
from filebackedarray import H5BackedSparseData

matrix = H5BackedSparseData("tests/data/tenx.sub.h5", group="matrix")

# get the dimensions of the matrix
print("matrix shape: ", matrix.shape)

# slice the matrix
matrix_slice = matrix[0:100, 1:101]
```

## Dense matrices

by default the matrix is assumed to be stored in C-style (row-major format). If the h5 file stores the matrix in a column-major format (Fortran-style), you can specify the `order="F"` parameter.

```python
from filebackedarray import H5BackedDenseData

matrix = H5BackedDenseData("tests/data/dense.h5", group="dense_C")

# get the dimensions of the matrix
print("matrix shape: ", matrix.shape)

# slice the matrix
matrix_slice = matrix[0:10, 1:10]
```