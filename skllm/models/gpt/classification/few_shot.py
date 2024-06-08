from skllm.models._base.classifier import (
    BaseFewShotClassifier,
    BaseDynamicFewShotClassifier,
    SingleLabelMixin,
    MultiLabelMixin,
)
from skllm.llm.gpt.mixin import GPTClassifierMixin
from skllm.models.gpt.vectorization import GPTVectorizer
from skllm.models._base.vectorizer import BaseVectorizer
from skllm.memory.base import IndexConstructor
from typing import Optional


class FewShotGPTClassifier(BaseFewShotClassifier, GPTClassifierMixin, SingleLabelMixin):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
        **kwargs,
    ):
        """
        Few-shot text classifier using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key, org)


class MultiLabelFewShotGPTClassifier(
    BaseFewShotClassifier, GPTClassifierMixin, MultiLabelMixin
):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
        **kwargs,
    ):
        """
        Multi-label few-shot text classifier using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        max_labels : Optional[int], optional
            maximum labels per sample, by default 5
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        """
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key, org)


class DynamicFewShotGPTClassifier(
    BaseDynamicFewShotClassifier, GPTClassifierMixin, SingleLabelMixin
):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_examples: int = 3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        """
        Dynamic few-shot text classifier using OpenAI/GPT API-compatible models.
        For each sample, N closest examples are retrieved from the memory.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        n_examples : int, optional
            number of closest examples per class to be retrieved, by default 3
        memory_index : Optional[IndexConstructor], optional
            custom memory index, for details check `skllm.memory` submodule, by default None
        vectorizer : Optional[BaseVectorizer], optional
            scikit-llm vectorizer; if None, `GPTVectorizer` is used, by default None
        metric : Optional[str], optional
            metric used for similarity search, by default "euclidean"
        """
        if vectorizer is None:
            vectorizer = GPTVectorizer(model="text-embedding-ada-002", key=key, org=org)
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
            metric=metric,
        )
        self._set_keys(key, org)
