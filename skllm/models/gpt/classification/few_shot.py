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
        **kwargs,
    ):
        if vectorizer is None:
            vectorizer = GPTVectorizer(model="text-embedding-ada-002")
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
        )
        self._set_keys(key, org)
