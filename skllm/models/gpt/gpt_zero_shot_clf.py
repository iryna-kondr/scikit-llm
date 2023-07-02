import random
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from skllm.models._base import _BaseZeroShotGPTClassifier
from skllm.prompts.builders import (
    build_zero_shot_prompt_mlc,
    build_zero_shot_prompt_slc,
)
from skllm.utils import extract_json_key


class ZeroShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """Zero-shot classifier for multiclass classification.

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
    default_label : Optional[str] , default : 'Random'
        The default label to use if the LLM could not generate a response for a sample. If set to 'Random' a random
        label will be chosen based on probabilities from the training set.
    """

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        default_label: Optional[str] = "Random",
    ):
        super().__init__(openai_key, openai_org, openai_model, default_label)

    def _get_prompt(self, x) -> str:
        return build_zero_shot_prompt_slc(x, repr(self.classes_))

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        y = self._to_np(y)
        return super().fit(X, y)


class MultiLabelZeroShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """Zero-shot classifier for multilabel classification.

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
    default_label : Optional[Union[List[str], Literal['Random']] , default : 'Random'
        The default label to use if the LLM could not generate a response for a sample. If set to 'Random' a random
        label will be chosen based on probabilities from the training set.
    max_labels : int , default : 3
        The maximum number of labels to predict for each sample.
    """

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        default_label: Optional[Union[List[str], Literal["Random"]]] = "Random",
        max_labels: int = 3,
    ):
        super().__init__(openai_key, openai_org, openai_model, default_label)
        if max_labels < 2:
            raise ValueError("max_labels should be at least 2")
        if isinstance(default_label, str) and default_label != "Random":
            raise ValueError("default_label should be a list of strings or 'Random'")
        self.max_labels = max_labels

    def _extract_labels(self, y) -> List[str]:
        """Extracts the labels into a list.

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

    def _get_default_label(self):
        """Returns the default label based on the default_label argument."""
        result = []
        if isinstance(self.default_label, str) and self.default_label == "Random":
            for cls, probability in zip(self.classes_, self.probabilities_):
                coin_flip = random.choices([0, 1], [1 - probability, probability])[0]
                if coin_flip == 1:
                    result.append(cls)
        else:
            result = self.default_label

        return result

    def _predict_single(self, x):
        """Predicts the labels for a single sample."""
        completion = self._get_chat_completion(x)
        try:
            labels = extract_json_key(
                completion["choices"][0]["message"]["content"], "label"
            )
            if not isinstance(labels, list):
                labels = labels.split(",")
                labels = [l.strip() for l in labels]
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            labels = []

        labels = list(filter(lambda l: l in self.classes_, labels))
        if len(labels) == 0:
            labels = self._get_default_label()
        if labels is not None and len(labels) > self.max_labels:
            labels = labels[: self.max_labels - 1]
        return labels

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: List[List[str]],
    ):
        """Calls the parent fit method on input data.

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            Input array data
        y : List[List[str]]
            The labels.
        """
        return super().fit(X, y)
