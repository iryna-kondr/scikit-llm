from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from skllm.models._base import _BaseZeroShotGPTClassifier
from skllm.prompts.builders import build_few_shot_prompt_mlc, build_few_shot_prompt_slc
from skllm.utils import extract_json_key
from skllm.utils import to_numpy as _to_numpy

_TRAINING_SAMPLE_PROMPT_TEMPLATE = """
Sample input:
```{x}```

Sample target: {label}
"""


class FewShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """Few-shot single-label classifier."""

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        """Fits the model to the given data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            training data
        y : Union[np.ndarray, pd.Series, List[str]]
            training labels

        Returns
        -------
        FewShotGPTClassifier
            self
        """
        if not len(X) == len(y):
            raise ValueError("X and y must have the same length.")
        X = _to_numpy(X)
        y = _to_numpy(y)
        self.training_data_ = (X, y)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self

    def _get_prompt_template(self) -> str:
        """Returns the prompt template to use.

        Returns
        -------
        str
            prompt template
        """
        if self.prompt_template is None:
            return _TRAINING_SAMPLE_PROMPT_TEMPLATE
        return self.prompt_template

    def _get_prompt(self, x: str) -> str:
        """Generates the prompt for the given input.

        Parameters
        ----------
        x : str
            sample to classify

        Returns
        -------
        str
            final prompt
        """
        prompt_template = self._get_prompt_template()
        training_data = []
        for xt, yt in zip(*self.training_data_):
            training_data.append(
                prompt_template.format(x=xt, label=yt)
            )

        training_data_str = "\n".join(training_data)

        return build_few_shot_prompt_slc(
            x=x, training_data=training_data_str, labels=repr(self.classes_)
        )


class MultiLabelFewShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """Few-shot multi-label classifier."""

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

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: List[List[str]],
    ):
        """Fits the model to the given data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            training data
        y : Union[np.ndarray, pd.Series, List[str]]
            training labels

        Returns
        -------
        FewShotGPTClassifier
            self
        """
        if not len(X) == len(y):
            raise ValueError("X and y must have the same length.")
        X = _to_numpy(X)
        y = _to_numpy(y)
        self.training_data_ = (X, y)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self

    def _get_prompt(self, x) -> str:
        training_data = []
        for xt, yt in zip(*self.training_data_):
            training_data.append(
                _TRAINING_SAMPLE_PROMPT_TEMPLATE.format(x=xt, label=yt)
            )

        training_data_str = "\n".join(training_data)

        return build_few_shot_prompt_mlc(
            x=x,
            training_data=training_data_str,
            labels=repr(self.classes_),
            max_cats=self.max_labels,
        )

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
