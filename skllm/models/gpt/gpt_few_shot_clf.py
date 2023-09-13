from typing import List, Union

import numpy as np
import pandas as pd

from skllm.models._base import _BaseZeroShotGPTClassifier
from skllm.prompts.builders import build_few_shot_prompt_slc
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
