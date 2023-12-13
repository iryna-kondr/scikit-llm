from skllm.models._base.classifier import (
    BaseTunableClassifier as _BaseTunableClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from skllm.llm.vertex.mixin import (
    VertexClassifierMixin as _VertexClassifierMixin,
    VertexTunableMixin as _VertexTunableMixin,
)
from typing import Optional


class _TunableClassifier(
    _BaseTunableClassifier, _VertexClassifierMixin, _VertexTunableMixin
):
    pass


class VertexClassifier(_TunableClassifier, _SingleLabelMixin):
    def __init__(
        self,
        base_model: str = "text-bison@002",
        n_update_steps: int = 1,
        default_label: Optional[str] = "Random",
    ):
        self._set_hyperparametersI(base_model=base_model, n_update_steps=n_update_steps)
        super().__init__(
            model=None,
            default_label=default_label,
        )
