from skllm.models._base.classifier import (
    BaseFewShotClassifier,
    BaseDynamicFewShotClassifier,
    SingleLabelMixin,
    MultiLabelMixin,
)
from skllm.llm.anthropic.mixin import ClaudeClassifierMixin
from skllm.models.gpt.vectorization import GPTVectorizer
from skllm.models._base.vectorizer import BaseVectorizer
from skllm.memory.base import IndexConstructor
from typing import Optional
from model_constants import ANTHROPIC_CLAUDE_MODEL, OPENAI_EMBEDDING_MODEL


class FewShotClaudeClassifier(BaseFewShotClassifier, ClaudeClassifierMixin, SingleLabelMixin):
    """Few-shot text classifier using Anthropic's Claude API for single-label classification tasks."""

    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Few-shot text classifier using Anthropic's Claude API.

        Parameters
        ----------
        model : str, optional
            model to use
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class MultiLabelFewShotClaudeClassifier(
    BaseFewShotClassifier, ClaudeClassifierMixin, MultiLabelMixin
):
    """Few-shot text classifier using Anthropic's Claude API for multi-label classification tasks."""
    
    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Multi-label few-shot text classifier using Anthropic's Claude API.

        Parameters
        ----------
        model : str, optional
            model to use
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies
        max_labels : Optional[int], optional
            maximum labels per sample, by default 5
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config
        """
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class DynamicFewShotClaudeClassifier(
    BaseDynamicFewShotClassifier, ClaudeClassifierMixin, SingleLabelMixin
):
    """
    Dynamic few-shot text classifier using Anthropic's Claude API for 
    single-label classification tasks with dynamic example selection using GPT embeddings.
    """

    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        n_examples: int = 3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        """
        Dynamic few-shot text classifier using Anthropic's Claude API.
        For each sample, N closest examples are retrieved from the memory.

        Parameters
        ----------
        model : str, optional
            model to use
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config
        n_examples : int, optional
            number of closest examples per class to be retrieved, by default 3
        memory_index : Optional[IndexConstructor], optional
            custom memory index, for details check `skllm.memory` submodule
        vectorizer : Optional[BaseVectorizer], optional
            scikit-llm vectorizer; if None, `GPTVectorizer` is used
        metric : Optional[str], optional
            metric used for similarity search, by default "euclidean"
        """
        if vectorizer is None:
            vectorizer = GPTVectorizer(model=OPENAI_EMBEDDING_MODEL, key=key)
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
            metric=metric,
        )
        self._set_keys(key)
