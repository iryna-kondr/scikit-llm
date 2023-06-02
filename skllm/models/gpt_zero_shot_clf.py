import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from skllm.completions import get_chat_completion
from skllm.openai.chatgpt import construct_message, extract_json_key
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
from skllm.prompts.builders import (
    build_zero_shot_prompt_mlc,
    build_zero_shot_prompt_slc,
)
from skllm.utils import to_numpy as _to_numpy


class _BaseZeroShotGPTClassifier(ABC, BaseEstimator, ClassifierMixin, _OAIMixin):
    """
    A base class for zero-shot classification using GPT-3.

    Initialization Parameters
    ----------
    openai_key : str, optional
        The OPEN AI key to use. Defaults to None.
    openai_org : str, optional
        The OPEN AI organization ID to use. Defaults to None.
    openai_model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    """
    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model

    def _to_np(self, X):
        """
        Convert X to a numpy array.
        
        Parameters
        ----------
        X : Any
            The input data to convert to a numpy array.
        
        Returns
        -------
        np.ndarray
            The input data as a numpy array.
        """
        return _to_numpy(X)

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        X = self._to_np(X)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """
        Predict the class of each input.
        
        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.
        
        Returns
        -------
        List[str]
        """
        X = self._to_np(X)
        predictions = []
        for i in tqdm(range(len(X))):
            predictions.append(self._predict_single(X[i]))
        return predictions

    @abstractmethod
    def _extract_labels(self, y: Any) -> List[str]:
        pass

    def _get_unique_targets(self, y:Any):
        labels = self._extract_labels(y)

        counts = Counter(labels)

        total = sum(counts.values())

        classes, probs = [], []
        for l, c in counts.items():
            classes.append(l)
            probs.append(c / total)

        return classes, probs

    def _get_chat_completion(self, x):
        prompt = self._get_prompt(x)
        msgs = []
        msgs.append(construct_message("system", "You are a text classification model."))
        msgs.append(construct_message("user", prompt))
        completion = get_chat_completion(
            msgs, self._get_openai_key(), self._get_openai_org(), self.openai_model
        )
        return completion


class ZeroShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """
    A zero-shot classifier using GPT-3.
    """
    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
    ):
        super().__init__(openai_key, openai_org, openai_model)

    def _extract_labels(self, y: Any) -> List[str]:
        """
        Return the class labels as a list.

        Parameters
        ----------
        y : Any
        
        Returns
        -------
        List[str]
        """
        if isinstance(y, (pd.Series, np.ndarray)):
            labels = y.tolist()
        else:
            labels = y
        return labels

    def _get_prompt(self, x) -> str:
        return build_zero_shot_prompt_slc(x, repr(self.classes_))

    def _predict_single(self, x: str) -> str:
        """
        Predict the class for a single input.

        Parameters
        ----------
        x : str
            The input to predict the class of.
        
        Returns
        -------
        str
        """
        completion = self._get_chat_completion(x)
        try:
            if self.openai_model.startswith("gpt4all::"):
                label = str(
                    extract_json_key(
                        completion["choices"][0]["message"]["content"], "label"
                    )
                )
            else:
                label = str(
                    extract_json_key(completion.choices[0].message["content"], "label")
                )
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""

        if label not in self.classes_:
            label = label.replace("'", "").replace('"', "")
            if label not in self.classes_:  # try again
                label = random.choices(self.classes_, self.probabilities_)[0]
        return label

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        y = self._to_np(y)
        return super().fit(X, y)


class MultiLabelZeroShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """
    A zero-shot classifier using GPT-3 for multi-label classification.
    
    Initialization Parameters
    ----------
    openai_key : str, optional
        The OPEN AI key to use. Defaults to None.
    openai_org : str, optional
        The OPEN AI organization ID to use. Defaults to None.
    openai_model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    max_labels : int, optional
        The maximum number of labels to predict. Defaults to 3.
    """
    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        max_labels: int = 3,
    ):
        super().__init__(openai_key, openai_org, openai_model)
        if max_labels < 2:
            raise ValueError("max_labels should be at least 2")
        self.max_labels = max_labels

    def _extract_labels(self, y) -> List[str]:
        """
        Extract the labels into a list.

        Parameters
        ----------
        y : Any

        Returns
        -------
        List[str]
        """
        labels = []
        for l in y:
            for j in l:
                labels.append(j)

        return labels

    def _get_prompt(self, x) -> str:
        return build_zero_shot_prompt_mlc(x, repr(self.classes_), self.max_labels)

    def _predict_single(self, x) -> List[str]:
        """
        Predict the class for a single input.

        Parameters
        ----------
        x : str
            The input to predict the class of.
        
        Returns
        -------
        str
        """
        completion = self._get_chat_completion(x)
        try:
            labels = extract_json_key(completion.choices[0].message["content"], "label")
            if not isinstance(labels, list):
                raise RuntimeError("Invalid labels type, expected list")
        except Exception:
            labels = []

        labels = list(filter(lambda l: l in self.classes_, labels))

        if len(labels) > self.max_labels:
            labels = labels[: self.max_labels - 1]
        elif len(labels) < 1:
            labels = [random.choices(self.classes_, self.probabilities_)[0]]
        return labels

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: List[List[str]],
    ):
        """
        Calls the parent fit method.
        
        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            The input data.
        y : List[List[str]]
            The labels.
        
        """
        return super().fit(X, y)
