import delayedarray


def chunk_shape(x):
    grid = delayedarray.chunk_grid(x)
    return (grid.boundaries[0][0], grid.boundaries[1][0])
