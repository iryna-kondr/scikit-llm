from sklearn.base import (
    BaseEstimator as _BaseEstimator,
    TransformerMixin as _TransformerMixin,
)
from typing import Any, Optional, Union, List
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import pandas as pd
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
from skllm.openai.embeddings import get_embedding as _get_embedding
from skllm.utils import to_numpy as _to_numpy


class GPTVectorizer(_BaseEstimator, _TransformerMixin, _OAIMixin):
    """
    A class that uses OPEN AI embedding model that converts text to GPT embeddings.

    Parameters
    ----------
    openai_embedding_model : str
        The OPEN AI embedding model to use. Defaults to "text-embedding-ada-002".
    openai_key : str, optional
        The OPEN AI key to use. Defaults to None.
    openai_org : str, optional
        The OPEN AI organization to use. Defaults to None.
    """
    def __init__(
        self,
        openai_embedding_model: str = "text-embedding-ada-002",
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
    ):
        self.openai_embedding_model = openai_embedding_model
        self._set_keys(openai_key, openai_org)

    def fit(self, X: Any = None, y: Any = None, **kwargs) -> "GPTVectorizer":
        """
        Fit the GPTVectorizer to the data.
        This is modelled to function as the sklearn fit method

        Parameters
        ----------
        X : Any, optional
        y : Any, optional
        kwargs : dict, optional

        Returns
        -------
        self : GPTVectorizer
        """
        return self

    def transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]]) -> ndarray:
        """
        Transform a list of strings into a list of GPT embeddings.
        This is modelled to function as the sklearn transform meethod

        Parameters
        ----------
        X : np.ndarray, pd.Series, or list
            The list of strings to transform into GPT embeddings.
        
        Returns
        -------
        embeddings : np.ndarray
        """
        X = _to_numpy(X)
        embeddings = []
        for i in tqdm(range(len(X))):
            embeddings.append(
                _get_embedding(X[i], self._get_openai_key(), self._get_openai_org())
            )
        embeddings = np.asarray(embeddings)
        return embeddings

    def fit_transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]], y=None, **fit_params) -> ndarray:
        """
        Fit and transform a list of strings into a list of GPT embeddings.
        This is modelled to function as the sklearn fit_transform method

        Parameters
        ----------
        X : np.ndarray, pd.Series, or list
            The list of strings to transform into GPT embeddings.
        y : Any, optional

        Returns
        -------
        embeddings : np.ndarray
        """
        return self.fit(X, y).transform(X)
