from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from tqdm import tqdm

from skllm.openai.embeddings import get_embedding as _get_embedding
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
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
