from abc import ABC, abstractmethod
from typing import Any, List, Type

from numpy import ndarray


class _BaseMemoryIndex(ABC):
    @abstractmethod
    def add(self, id: Any, vector: ndarray):
        """Adds a vector to the index.

        Parameters
        ----------
        id : Any
            identifier for the vector
        vector : ndarray
            vector to add to the index
        """
        pass

    @abstractmethod
    def retrieve(self, vectors: ndarray, k: int) -> List:
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
        pass

    @abstractmethod
    def build(self) -> None:
        """Builds the index.

        All build parameters should be passed to the constructor.
        """
        pass


class IndexConstructor:
    def __init__(self, index: Type[_BaseMemoryIndex], **kwargs: Any) -> None:
        self.index = index
        self.kwargs = kwargs

    def __call__(self) -> _BaseMemoryIndex:
        return self.index(**self.kwargs)
