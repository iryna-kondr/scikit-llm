from typing import Optional, Union, List
import pandas as pd
from skllm.models._base import _BaseZeroShotGPTClassifier
from skllm.prompts.builders import build_zero_shot_prompt_slc
from skllm.openai.credentials import set_credentials
from skllm.openai.tuning import create_tuning_job, await_results, delete_file
import numpy as np
import json
import uuid


def _build_clf_example(
    x: str, y: str, system_msg="You are a text classification model."
):
    sample = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": x},
            {"role": "assistant", "content": y},
        ]
    }
    return json.dumps(sample)


class _Tunable:
    system_msg = "You are a text classification model."

    def _build_label(self, label: str):
        return json.dumps({"label": label})

    def _tune(self, X, y):
        file_uuid = str(uuid.uuid4())
        filename = f"skllm_{file_uuid}.jsonl"
        with open(filename, "w+") as f:
            for xi, yi in zip(X, y):
                f.write(
                    _build_clf_example(
                        self._get_prompt(xi), self._build_label(yi), self.system_msg
                    )
                )
                f.write("\n")
        set_credentials(self._get_openai_key(), self._get_openai_org())
        job = create_tuning_job(
            self.base_model,
            filename,
            self.n_epochs,
            self.custom_suffix,
        )
        print(f"Created new tuning job. JOB_ID = {job['id']}")
        job = await_results(job["id"])
        self.openai_model = job["fine_tuned_model"]
        delete_file(job["training_file"])
        print(f"Finished training. Number of trained tokens: {job['trained_tokens']}.")


class GPTClassifier(_BaseZeroShotGPTClassifier, _Tunable):
    """Fine-tunable GPT classifier for single-label classification."""

    supported_models = ["gpt-3.5-turbo-0613"]

    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        default_label: Optional[str] = "Random",
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
    ):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.custom_suffix = custom_suffix
        if base_model not in self.supported_models:
            raise ValueError(
                f"Model {base_model} is not supported. Supported models are"
                f" {self.supported_models}"
            )
        super().__init__(
            openai_model="undefined",
            default_label=default_label,
            openai_key=openai_key,
            openai_org=openai_org,
        )

    def _get_prompt(self, x: str) -> str:
        return build_zero_shot_prompt_slc(x, repr(self.classes_))

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
        GPTClassifier
            self
        """
        X = self._to_np(X)
        y = self._to_np(y)
        super().fit(X, y)
        self._tune(X, y)
        return self


# similarly to PaLM, this is not a classifier, but a quick way to re-use the code
# the hierarchy of classes will be reworked in the next releases
class GPT(_BaseZeroShotGPTClassifier, _Tunable):
    """Fine-tunable GPT on arbitrary input-output pairs."""

    supported_models = ["gpt-3.5-turbo-0613"]

    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
        system_msg: Optional[str] = "You are a text processing model.",
    ):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.custom_suffix = custom_suffix
        self.system_msg = system_msg
        if base_model not in self.supported_models:
            raise ValueError(
                f"Model {base_model} is not supported. Supported models are"
                f" {self.supported_models}"
            )
        super().__init__(
            openai_model="undefined",  # this will be rewritten later
            default_label="Random",  # just for compatibility
            openai_key=openai_key,
            openai_org=openai_org,
        )

    def _get_prompt(self, x: str) -> str:
        return x

    def _build_label(self, label: str):
        return label

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
        GPT
            self
        """
        X = self._to_np(X)
        y = self._to_np(y)
        self._tune(X, y)
        return self
