from typing import List, Union

import numpy as np
import pandas as pd

from skllm.google.completions import get_completion
from skllm.google.tuning import tune
from skllm.models._base import _BasePaLMClassifier


# This is not a classifier, but inherits from the base classifier anyway for simplicity reasons.
class PaLM(_BasePaLMClassifier):
    """Google PaLM for single-label classification.

    Parameters
    ----------
    model : str
        The name of the model to use. Must be one of the models supported by google. Defaults to "text-bison@001".
    n_update_steps : int, optional
        The number of update steps to perform during tuning. Defaults to 1.
    """

    _supported_models = ["text-bison@001"]

    def __init__(self, model="text-bison@001", n_update_steps: int = 1) -> None:
        super().__init__(model, default_label=None)
        if model not in self._supported_models:
            raise ValueError(
                f"Model {model} not supported. Supported models:"
                f" {self._supported_models}"
            )
        self.n_update_steps = int(n_update_steps)

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        """Fits the model to the data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            Inputs
        y : Union[np.ndarray, pd.Series, List[str]]
            Targets
        """
        X = self._to_np(X)
        y = self._to_np(y)
        if len(X) != len(y):
            raise ValueError(
                f"X and y must be the same length. Got {len(X)} and {len(y)}"
            )
        if len(X) < 10:
            raise ValueError(
                f"The dataset must have at least 10 datapoints. Got {len(X)}."
            )
        df = pd.DataFrame({"input_text": X, "output_text": y})
        job = tune(self.model, df, self.n_update_steps)._job
        tuned_model = job.result()
        self.tuned_model_ = tuned_model._model_resource_name
        return self

    def _predict_single(self, x: str) -> str:
        label = str(get_completion(self.tuned_model_, x))
        return label

    def _get_prompt(self, x: str) -> str:
        """For compatibility with the parent class."""
        return x
