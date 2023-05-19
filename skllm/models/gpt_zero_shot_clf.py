from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from skllm.openai.prompts import get_zero_shot_prompt_slc, get_zero_shot_prompt_mlc
from skllm.openai.chatgpt import (
    construct_message,
    get_chat_completion,
    extract_json_key,
)
from skllm.config import SKLLMConfig as _Config
from skllm.utils import to_numpy as _to_numpy
from skllm.openai.mixin import OpenAIMixin as _OAIMixin

class _BaseZeroShotGPTClassifier(ABC, BaseEstimator, ClassifierMixin, _OAIMixin):
    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model

    def _to_np(self, X):
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
        X = self._to_np(X)
        predictions = []
        for i in tqdm(range(len(X))):
            predictions.append(self._predict_single(X[i]))
        return predictions

    @abstractmethod
    def _extract_labels(self, y: Any) -> List[str]:
        pass

    def _get_unique_targets(self, y):
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
    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
    ):
        super().__init__(openai_key, openai_org, openai_model)

    def _extract_labels(self, y: Any) -> List[str]:
        if isinstance(y, (pd.Series, np.ndarray)):
            labels = y.tolist()
        else:
            labels = y
        return labels

    def _get_prompt(self, x) -> str:
        return get_zero_shot_prompt_slc(x, self.classes_)

    def _predict_single(self, x):
        completion = self._get_chat_completion(x)
        try:
            label = str(
                extract_json_key(completion.choices[0].message["content"], "label")
            )
        except Exception as e:
            label = ""

        if label not in self.classes_:
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
        labels = []
        for l in y:
            for j in l:
                labels.append(j)

        return labels

    def _get_prompt(self, x) -> str:
        return get_zero_shot_prompt_mlc(x, self.classes_, self.max_labels)

    def _predict_single(self, x):
        completion = self._get_chat_completion(x)
        try:
            labels = extract_json_key(completion.choices[0].message["content"], "label")
            if not isinstance(labels, list):
                raise RuntimeError("Invalid labels type, expected list")
        except Exception as e:
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
        return super().fit(X, y)
