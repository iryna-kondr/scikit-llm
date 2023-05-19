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
    def __init__(
        self,
        openai_embedding_model: str = "text-embedding-ada-002",
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
    ):
        self.openai_embedding_model = openai_embedding_model
        self._set_keys(openai_key, openai_org)

    def fit(self, X: Any = None, y: Any = None, **kwargs):
        return self

    def transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]]) -> ndarray:
        X = _to_numpy(X)
        embeddings = []
        for i in tqdm(range(len(X))):
            embeddings.append(
                _get_embedding(X[i], self._get_openai_key(), self._get_openai_org())
            )
        embeddings = np.asarray(embeddings)
        return embeddings

    def fit_transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]], y=None, **fit_params) -> ndarray:
        return self.fit(X, y).transform(X)
