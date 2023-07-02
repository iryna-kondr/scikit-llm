import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from skllm.completions import get_chat_completion
from skllm.openai.chatgpt import construct_message
from skllm.openai.mixin import OpenAIMixin as _OAIMixin
from skllm.utils import extract_json_key
from skllm.utils import to_numpy as _to_numpy


class BaseClassifier(ABC, BaseEstimator, ClassifierMixin):
    default_label: Optional[str] = "Random"

    def _to_np(self, X):
        """Converts X to a numpy array.

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

    @abstractmethod
    def _predict_single(self, x: str) -> Any:
        """Predicts the class of a single input."""
        pass

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        """Extracts the target for each datapoint in X.

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            The input array data to fit the model to.

        y : Union[np.ndarray, pd.Series, List[str], List[List[str]]]
            The target array data to fit the model to.
        """
        X = self._to_np(X)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """Predicts the class of each input.

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

    def _get_unique_targets(self, y: Any):
        labels = self._extract_labels(y)

        counts = Counter(labels)

        total = sum(counts.values())

        classes, probs = [], []
        for l, c in counts.items():
            classes.append(l)
            probs.append(c / total)

        return classes, probs

    def _extract_labels(self, y: Any) -> List[str]:
        """Return the class labels as a list.

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

    def _get_default_label(self):
        """Returns the default label based on the default_label argument."""
        if self.default_label == "Random":
            return random.choices(self.classes_, self.probabilities_)[0]
        else:
            return self.default_label


class _BaseZeroShotGPTClassifier(BaseClassifier, _OAIMixin):
    """Base class for zero-shot classifiers.

    Parameters
    ----------
    openai_key : Optional[str] , default : None
        Your OpenAI API key. If None, the key will be read from the SKLLM_CONFIG_OPENAI_KEY environment variable.
    openai_org : Optional[str] , default : None
        Your OpenAI organization. If None, the organization will be read from the SKLLM_CONFIG_OPENAI_ORG
         environment variable.
    openai_model : str , default : "gpt-3.5-turbo"
        The OpenAI model to use. See https://beta.openai.com/docs/api-reference/available-models for a list of
        available models.
    default_label : Optional[Union[List[str], str]] , default : 'Random'
        The default label to use if the LLM could not generate a response for a sample. If set to 'Random' a random
        label will be chosen based on probabilities from the training set.
    """

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        default_label: Optional[Union[List[str], str]] = "Random",
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model
        self.default_label = default_label

    @abstractmethod
    def _get_prompt(self, x: str) -> str:
        """Generates a prompt for the given input."""
        pass

    def _get_chat_completion(self, x):
        prompt = self._get_prompt(x)
        msgs = []
        msgs.append(construct_message("system", "You are a text classification model."))
        msgs.append(construct_message("user", prompt))
        completion = get_chat_completion(
            msgs, self._get_openai_key(), self._get_openai_org(), self.openai_model
        )
        return completion

    def _predict_single(self, x):
        """Predicts the labels for a single sample.

        Should work for all (single label) GPT based classifiers.
        """
        completion = self._get_chat_completion(x)
        try:
            label = str(
                extract_json_key(
                    completion["choices"][0]["message"]["content"], "label"
                )
            )
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""

        if label not in self.classes_:
            label = label.replace("'", "").replace('"', "")
            if label not in self.classes_:  # try again
                label = self._get_default_label()
        return label


class _BasePaLMClassifier(BaseClassifier):
    def __init__(self, model: str, default_label: Optional[str] = "Random"):
        self.model = model
        self.default_label = default_label
