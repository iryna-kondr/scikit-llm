from typing import Any, List

from annoy import AnnoyIndex
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

    def __init__(self, dim: int, metric: str = "euclidean", **kwargs: Any) -> None:
        self._index = AnnoyIndex(dim, metric)
        self.built = False

    def add(self, id: int, vector: ndarray) -> None:
        """Adds a vector to the index.

        Parameters
        ----------
        id : Any
            identifier for the vector
        vector : ndarray
            vector to add to the index
        """
        if self.built:
            raise RuntimeError("Cannot add vectors after index is built.")
        self._index.add_item(id, vector)

    def build(self) -> None:
        """Builds the index.

        No new vectors can be added after building.
        """
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
