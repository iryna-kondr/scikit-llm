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
from skllm.prompts.builders import (
    build_focused_summary_prompt,
    build_summary_prompt,
    build_translation_prompt,
)


class BaseText2TextModel(ABC, _SklBaseEstimator, _SklTransformerMixin):
    def fit(self, X: Any, y: Any = None):
        """
        Fits the model to the data. Usually a no-op.

        Parameters
        ----------
        X : Any
            training data
        y : Any
            training outputs

        Returns
        -------
        self
            BaseText2TextModel
        """
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        return self.transform(X)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Optional[Union[np.ndarray, pd.Series, List[str]]] = None,
    ) -> ndarray:
        return self.fit(X, y).transform(X)

    def transform(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """
        Transforms the input data.

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
        prediction = self._convert_completion_to_str(prediction)
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
        """
        Fits the model to the data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            training data
        y : Union[np.ndarray, pd.Series, List[str]]
            training labels

        Returns
        -------
        BaseTunableText2TextModel
            self
        """
        if not isinstance(self, _BaseTunableMixin):
            raise TypeError(
                "Classifier must be mixed with a skllm.llm.base.BaseTunableMixin class"
            )
        self._tune(X, y)
        return self

    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        return {"messages": str(x), "system_message": ""}

    def _predict_single(self, x: str) -> str:
        if self.model is None:
            raise RuntimeError("Model has not been tuned yet")
        return super()._predict_single(x)


class BaseSummarizer(BaseText2TextModel):
    max_words: int = 15
    focus: Optional[str] = None
    system_message: str = "You are a text summarizer."

    def _get_prompt(self, X: str) -> str:
        if self.focus:
            prompt = build_focused_summary_prompt(X, self.max_words, self.focus)
        else:
            prompt = build_summary_prompt(X, self.max_words)
        return {"messages": prompt, "system_message": self.system_message}

    def transform(
        self, X: Union[ndarray, pd.Series, List[str]], **kwargs: Any
    ) -> ndarray:
        """
        Transforms the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.

        Returns
        -------
        List[str]
        """
        y = super().transform(X, **kwargs)
        if self.focus:
            y = np.asarray(
                [
                    i.replace("Mentioned concept is not present in the text.", "")
                    .replace("The general summary is:", "")
                    .strip()
                    for i in y
                ],
                dtype=object,
            )
        return y


class BaseTranslator(BaseText2TextModel):
    output_language: str = "English"
    system_message = "You are a text translator."

    def _get_prompt(self, X: str) -> str:
        prompt = build_translation_prompt(X, self.output_language)
        return {"messages": prompt, "system_message": self.system_message}

    def transform(
        self, X: Union[ndarray, pd.Series, List[str]], **kwargs: Any
    ) -> ndarray:
        """
        Transforms the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.

        Returns
        -------
        List[str]
        """
        y = super().transform(X, **kwargs)
        y = np.asarray(
            [i.replace("[Translated text:]", "").replace("```", "").strip() for i in y],
            dtype=object,
        )
        return y
