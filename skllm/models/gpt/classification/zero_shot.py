from skllm.models._base.classifier import (
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
)
from skllm.llm.gpt.mixin import GPTClassifierMixin as _GPTClassifierMixin
from typing import Optional


class ZeroShotGPTClassifier(
    _BaseZeroShotClassifier, _GPTClassifierMixin, _SingleLabelMixin
):
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


class MultiLabelZeroShotGPTClassifier(
    _BaseZeroShotClassifier, _GPTClassifierMixin, _MultiLabelMixin
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
