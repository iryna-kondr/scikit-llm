from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from tqdm import tqdm

from skllm.openai.embeddings import get_embedding
from skllm.openai.mixin import OpenAIMixin
from skllm.utils import to_numpy


class GPTVectorizer(BaseEstimator, TransformerMixin, OpenAIMixin):
    def __init__(
        self,
        openai_embedding_model: str = "text-embedding-ada-002",
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.openai_embedding_model = openai_embedding_model
        self._set_keys(openai_key, openai_org)
        self.batch_size = batch_size

    def fit(self, X: Any = None, y: Any = None, **kwargs: Any):
        return self

    def transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]]) -> ndarray:
        X = to_numpy(X)
        num_samples = len(X)
        embeddings = []

        def process_batch(batch_texts):
            return get_embedding(
                batch_texts,
                self._get_openai_key(),
                self._get_openai_org(),
                model=self.openai_embedding_model,
            )

        for i in tqdm(range(0, num_samples, self.batch_size)):
            batch_texts = X[i : i + self.batch_size]
            batch_embeddings = Parallel(n_jobs=-1)(
                delayed(process_batch)(texts) for texts in batch_texts
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.asarray(embeddings)
        return embeddings

    def fit_transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]], y=None, **fit_params: Any) -> ndarray:
        return self.fit(X, y).transform(X)
