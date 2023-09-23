from skllm.llm.vertex.mixin import VertexClassifierMixin as _VertexClassifierMixin
from skllm.models._base.classifier import (
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from typing import Optional


class ZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _SingleLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = "text-bison@001",
        default_label: Optional[str] = "Random",
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )


class MultiLabelZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _MultiLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = "text-bison@001",
        default_label: Optional[str] = "Random",
        prompt_template: Optional[str] = None,
        max_labels: Optional[int] = 5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            max_labels=max_labels,
            **kwargs,
        )
