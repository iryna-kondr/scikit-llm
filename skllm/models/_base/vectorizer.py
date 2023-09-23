from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from skllm.utils import to_numpy as _to_numpy
from skllm.llm.base import BaseEmbeddingMixin
from sklearn.base import (
    BaseEstimator as _SklBaseEstimator,
    TransformerMixin as _SklTransformerMixin,
)


class BaseVectorizer(_SklBaseEstimator, _SklTransformerMixin):
    """
    A base vectorization/embedding class.

    Parameters
    ----------
    model : str
        The embedding model to use.
    """

    def __init__(self, model: str, batch_size: int = 1):
        if not isinstance(self, BaseEmbeddingMixin):
            raise TypeError(
                "Vectorizer must be mixed with skllm.llm.base.BaseEmbeddingMixin."
            )
        self.model = model
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size

    def fit(self, X: Any = None, y: Any = None, **kwargs):
        """
        Does nothing. Needed only for sklearn compatibility.

        Parameters
        ----------
        X : Any, optional
        y : Any, optional
        kwargs : dict, optional

        Returns
        -------
        self : BaseVectorizer
        """
        return self

    def transform(
        self, X: Optional[Union[np.ndarray, pd.Series, List[str]]]
    ) -> np.ndarray:
        """
        Transforms a list of strings into a list of GPT embeddings.
        This is modelled to function as the sklearn transform method

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            The input array of strings to transform into GPT embeddings.

        Returns
        -------
        embeddings : np.ndarray
        """
        X = _to_numpy(X)
        embeddings = self._get_embeddings(X)
        embeddings = np.asarray(embeddings)
        return embeddings

    def fit_transform(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Any = None,
        **fit_params,
    ) -> np.ndarray:
        """
        Fits and transforms a list of strings into a list of embeddings.
        This is modelled to function as the sklearn fit_transform method

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            The input array of strings to transform into embeddings.
        y : Any, optional

        Returns
        -------
        embeddings : np.ndarray
        """
        return self.fit(X, y).transform(X)
