from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from tqdm import tqdm

from skllm.openai.chatgpt import construct_message, get_chat_completion
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
from skllm.utils import to_numpy as _to_numpy


class BaseZeroShotGPTTransformer(_BaseEstimator, _TransformerMixin, _OAIMixin):
    system_msg = "You are an scikit-learn transformer."
    default_output = "Output is unavailable"

    def _get_chat_completion(self, X):
        """Gets the chat completion for the given input using open ai API.

        Parameters
        ----------
        X : str
            Input string

        Returns
        -------
        str
        """
        prompt = self._get_prompt(X)
        msgs = []
        msgs.append(construct_message("system", self.system_msg))
        msgs.append(construct_message("user", prompt))
        completion = get_chat_completion(
            msgs, self._get_openai_key(), self._get_openai_org(), self.openai_model
        )
        try:
            return completion.choices[0].message["content"]
        except Exception as e:
            print(f"Skipping a sample due to the following error: {str(e)}")
            return self.default_output

    def fit(
        self, X: Any = None, y: Any = None, **kwargs: Any
    ) -> BaseZeroShotGPTTransformer:
        """Fits the model to the data.

        Parameters
        ----------
        X : Any, optional
        y : Any, optional
        kwargs : dict, optional

        Returns
        -------
        self : BaseZeroShotGPTTransformer
        """
        return self

    def transform(
        self, X: np.ndarray | pd.Series | list[str], **kwargs: Any
    ) -> ndarray:
        """Converts a list of strings using the open ai API and a predefined
        prompt.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]

        Returns
        -------
        ndarray
        """
        X = _to_numpy(X)
        transformed = []
        for i in tqdm(range(len(X))):
            transformed.append(self._get_chat_completion(X[i]))
        transformed = np.asarray(transformed, dtype=object)
        return transformed

    def fit_transform(
        self, X: np.ndarray | pd.Series | list[str], y=None, **fit_params
    ) -> ndarray:
        """Fits and transforms a list of strings using the transform method.
        This is modelled to function as the sklearn fit_transform method.

        Parameters
        ----------
        X : np.ndarray, pd.Series, or list

        Returns
        -------
        ndarray
        """
        return self.fit(X, y).transform(X)
