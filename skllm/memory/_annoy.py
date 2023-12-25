import os
import tempfile
from typing import Any, List

try:
    from annoy import AnnoyIndex
except (ImportError, ModuleNotFoundError):
    AnnoyIndex = None

from numpy import ndarray

from skllm.memory.base import _BaseMemoryIndex


class AnnoyMemoryIndex(_BaseMemoryIndex):
    """Memory index using Annoy.

    Parameters
    ----------
    dim : int
        dimensionality of the vectors
    metric : str, optional
        metric to use, by default "euclidean"
    """

    def __init__(self, dim: int = -1, metric: str = "euclidean", **kwargs: Any) -> None:
        if AnnoyIndex is None:
            raise ImportError(
                "Annoy is not installed. Please install annoy by running `pip install"
                " scikit-llm[annoy]`."
            )
        self.metric = metric
        self.dim = dim
        self.built = False
        self._index = None
        self._counter = 0

    def add(self, vector: ndarray) -> None:
        """Adds a vector to the index.

        Parameters
        ----------
        vector : ndarray
            vector to add to the index
        """
        if self.built:
            raise RuntimeError("Cannot add vectors after index is built.")
        if self.dim < 0:
            raise ValueError("Dimensionality must be positive.")
        if not self._index:
            self._index = AnnoyIndex(self.dim, self.metric)
        id = self._counter
        self._index.add_item(id, vector)
        self._counter += 1

    def build(self) -> None:
        """Builds the index.

        No new vectors can be added after building.
        """
        if self.dim < 0:
            raise ValueError("Dimensionality must be positive.")
        self._index.build(-1)
        self.built = True

    def retrieve(self, vectors: ndarray, k: int) -> List[List[int]]:
        """Retrieves the k nearest neighbors for each vector.

        Parameters
        ----------
        vectors : ndarray
            vectors to retrieve nearest neighbors for
        k : int
            number of nearest neighbors to retrieve

        Returns
        -------
        List
            ids of retrieved nearest neighbors
        """
        if not self.built:
            raise RuntimeError("Cannot retrieve vectors before the index is built.")
        return [
            self._index.get_nns_by_vector(v, k, search_k=-1, include_distances=False)
            for v in vectors
        ]

    def __getstate__(self) -> dict:
        """Returns the state of the object. To store the actual annoy index, it
        has to be written to a temporary file.

        Returns
        -------
        dict
            state of the object
        """
        state = self.__dict__.copy()

        # save index to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_filename = tmp.name
            self._index.save(temp_filename)

        # read bytes from the file
        with open(temp_filename, "rb") as tmp:
            index_bytes = tmp.read()

        # store bytes representation in state
        state["_index"] = index_bytes

        # remove temporary file
        os.remove(temp_filename)

        return state

    def __setstate__(self, state: dict) -> None:
        """Sets the state of the object. It restores the annoy index from the
        bytes representation.

        Parameters
        ----------
        state : dict
            state of the object
        """
        self.__dict__.update(state)
        # restore index from bytes
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_filename = tmp.name
            tmp.write(self._index)

        self._index = AnnoyIndex(self.dim, self.metric)
        self._index.load(temp_filename)

        # remove temporary file
        os.remove(temp_filename)
