from typing import Any, Union, List, Optional
from abc import abstractmethod, ABC
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import pandas as pd
from skllm.utils import to_numpy as _to_numpy
from sklearn.base import (
    BaseEstimator as _SklBaseEstimator,
    TransformerMixin as _SklTransformerMixin,
)
from skllm.llm.base import BaseTunableMixin as _BaseTunableMixin


class BaseText2TextModel(ABC, _SklBaseEstimator, _SklTransformerMixin):
    def fit(self, X: Any, y: Any):
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        return self.transform(X)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ) -> ndarray:
        return self.fit(X, y).transform(X)

    def transform(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """Predicts the class of each input.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.

        Returns
        -------
        List[str]
        """
        X = _to_numpy(X)
        predictions = []
        for i in tqdm(range(len(X))):
            predictions.append(self._predict_single(X[i]))
        return predictions

    def _predict_single(self, x: Any) -> Any:
        prompt_dict = self._get_prompt(x)
        # this will be inherited from the LLM
        prediction = self._get_chat_completion(model=self.model, **prompt_dict)
        return prediction

    @abstractmethod
    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        pass


class BaseTunableText2TextModel(BaseText2TextModel):
    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        if not isinstance(self, _BaseTunableMixin):
            raise TypeError(
                "Classifier must be mixed with a skllm.llm.base.BaseTunableMixin class"
            )
        self._tune(X, y)
        return self

    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        return str(x) 

    def _predict_single(self, x: str) -> str:
        if self.model is None:
            raise RuntimeError("Model has not been tuned yet")
        return super()._predict_single(x)
