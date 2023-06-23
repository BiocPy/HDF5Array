__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class BackedArray:
    """Base class for all file backed arrays."""

    def __init__(self) -> None:
        self._transpose = False

    @property
    def transposed(self) -> bool:
        """Is the array transposed?

        Returns:
            bool: returns True if array is transposed.
        """
        return self._transpose

    @transposed.setter
    def transposed(self, transposed: bool):
        if not isinstance(transposed, bool):
            raise ValueError("transposed must be a boolean")

        self._transpose = transposed
