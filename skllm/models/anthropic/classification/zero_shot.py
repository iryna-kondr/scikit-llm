from skllm.models._base.classifier import (
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
    BaseCoTClassifier as _BaseCoTClassifier,
)
from skllm.llm.anthropic.mixin import ClaudeClassifierMixin as _ClaudeClassifierMixin
from typing import Optional
from model_constants import ANTHROPIC_CLAUDE_MODEL


class ZeroShotClaudeClassifier(
    _BaseZeroShotClassifier, _ClaudeClassifierMixin, _SingleLabelMixin
):
    """Zero-shot text classifier using Anthropic Claude models for single-label classification."""
    
    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Zero-shot text classifier using Anthropic Claude models.

        Parameters
        ----------
        model : str, optional
            Model to use
        default_label : str, optional
            Default label for failed predictions; if "Random", selects randomly based on class frequencies, defaults to "Random".
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class CoTClaudeClassifier(
    _BaseCoTClassifier, _ClaudeClassifierMixin, _SingleLabelMixin
):
    """Chain-of-thought text classifier using Anthropic Claude models for single-label classification."""
    
    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Chain-of-thought text classifier using Anthropic Claude models.

        Parameters
        ----------
        model : str, optional
            Model to use.
        default_label : str, optional
            Default label for failed predictions; if "Random", selects randomly based on class frequencies, defaults to "Random".
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class MultiLabelZeroShotClaudeClassifier(
    _BaseZeroShotClassifier, _ClaudeClassifierMixin, _MultiLabelMixin
):
    """Zero-shot text classifier using Anthropic Claude models for multi-label classification."""
    
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
        Multi-label zero-shot text classifier using Anthropic Claude models.

        Parameters
        ----------
        model : str, optional
            Model to use.
        default_label : str, optional
            Default label for failed predictions; if "Random", selects randomly based on class frequencies, defaults to "Random".
        max_labels : Optional[int], optional
            Maximum number of labels per sample, by default 5.
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)