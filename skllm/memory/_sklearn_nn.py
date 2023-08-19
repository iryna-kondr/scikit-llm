from typing import Any, List

import numpy as np
from sklearn.neighbors import NearestNeighbors

from skllm.memory.base import _BaseMemoryIndex


class SklearnMemoryIndex(_BaseMemoryIndex):
    """Memory index using Sklearn's NearestNeighbors.

    Parameters
    ----------
    dim : int
        dimensionality of the vectors
    metric : str, optional
        metric to use, by default "euclidean"
    """

    def __init__(self, dim: int = -1, metric: str = "euclidean", **kwargs: Any) -> None:
        self._index = NearestNeighbors(metric=metric, **kwargs)
        self.metric = metric
        self.dim = dim
        self.built = False
        self.data = []

    def add(self, vector: np.ndarray) -> None:
        """Adds a vector to the index.

        Parameters
        ----------
        vector : np.ndarray
            vector to add to the index
        """
        if self.built:
            raise RuntimeError("Cannot add vectors after index is built.")
        self.data.append(vector)

    def build(self) -> None:
        """Builds the index.

        No new vectors can be added after building.
        """
        data_matrix = np.array(self.data)
        self._index.fit(data_matrix)
        self.built = True

    def retrieve(self, vectors: np.ndarray, k: int) -> List[List[int]]:
        """Retrieves the k nearest neighbors for each vector.

        Parameters
        ----------
        vectors : np.ndarray
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
        _, indices = self._index.kneighbors(vectors, n_neighbors=k)
        return indices.tolist()
