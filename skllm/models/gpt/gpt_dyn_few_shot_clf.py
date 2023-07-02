from __future__ import annotations

import numpy as np
import pandas as pd

from skllm.memory import AnnoyMemoryIndex
from skllm.models._base import _BaseZeroShotGPTClassifier
from skllm.preprocessing import GPTVectorizer
from skllm.prompts.builders import build_few_shot_prompt_slc
from skllm.utils import to_numpy

_TRAINING_SAMPLE_PROMPT_TEMPLATE = """
Sample input:
```{x}```

Sample target: {label}
"""


class DynamicFewShotGPTClassifier(_BaseZeroShotGPTClassifier):
    """Dynamic few-shot single-label classifier.

    Parameters
    ----------
    n_examples : int, optional
        number of examples per class, by default 3
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
        n_examples: int = 3,
        openai_key: str | None = None,
        openai_org: str | None = None,
        openai_model: str = "gpt-3.5-turbo",
        default_label: str | None = "Random",
    ):
        super().__init__(openai_key, openai_org, openai_model, default_label)
        self.n_examples = n_examples

    def fit(
        self,
        X: np.ndarray | pd.Series | list[str],
        y: np.ndarray | pd.Series | list[str],
    ) -> DynamicFewShotGPTClassifier:
        """Fits the model to the given data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            training data
        y : Union[np.ndarray, pd.Series, List[str]]
            training labels

        Returns
        -------
        DynamicFewShotGPTClassifier
            self
        """
        X = to_numpy(X)
        y = to_numpy(y)
        self.embedding_model_ = GPTVectorizer().fit(X)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)

        self.data_ = {}
        for cls in self.classes_:
            print(f"Building index for class `{cls}` ...")
            self.data_[cls] = {}
            partition = X[y == cls]
            self.data_[cls]["partition"] = partition
            embeddings = self.embedding_model_.transform(partition)
            index = AnnoyMemoryIndex(embeddings.shape[1])
            for i, embedding in enumerate(embeddings):
                index.add(i, embedding)
            index.build()
            self.data_[cls]["index"] = index

        return self

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
        embedding = self.embedding_model_.transform([x])
        training_data = []
        for cls in self.classes_:
            index = self.data_[cls]["index"]
            partition = self.data_[cls]["partition"]
            neighbors = index.retrieve(embedding, min(self.n_examples, len(partition)))
            neighbors = [partition[i] for i in neighbors[0]]
            training_data.extend(
                [
                    _TRAINING_SAMPLE_PROMPT_TEMPLATE.format(x=neighbor, label=cls)
                    for neighbor in neighbors
                ]
            )

        training_data_str = "\n".join(training_data)

        return build_few_shot_prompt_slc(
            x=x, training_data=training_data_str, labels=repr(self.classes_)
        )
