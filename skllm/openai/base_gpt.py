import numpy as np
from numpy import ndarray
import pandas as pd
from skllm.utils import to_numpy as _to_numpy
from typing import Any, Optional, Union, List
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
from tqdm import tqdm
from sklearn.base import (
    BaseEstimator as _BaseEstimator,
    TransformerMixin as _TransformerMixin,
)
from skllm.openai.chatgpt import (
    construct_message,
    get_chat_completion,
)

class BaseZeroShotGPTTransformer(_BaseEstimator, _TransformerMixin, _OAIMixin): 
    
    system_msg = "You are an scikit-learn transformer."
    default_output = "Output is unavailable"

    def _get_chat_completion(self, X):
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
            
    def fit(self, X: Any = None, y: Any = None, **kwargs: Any):
        return self

    def transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]], **kwargs: Any) -> ndarray:
        X = _to_numpy(X)
        transformed = []
        for i in tqdm(range(len(X))):
            transformed.append(
                self._get_chat_completion(X[i])
            )
        transformed = np.asarray(transformed, dtype = object)
        return transformed

    def fit_transform(self, X: Optional[Union[np.ndarray, pd.Series, List[str]]], y=None, **fit_params) -> ndarray:
        return self.fit(X, y).transform(X)