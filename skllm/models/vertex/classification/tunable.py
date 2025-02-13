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
from model_constants import TEXT_BISON_MODEL


class _TunableClassifier(
    _BaseTunableClassifier, _VertexClassifierMixin, _VertexTunableMixin
):
    pass


class VertexClassifier(_TunableClassifier, _SingleLabelMixin):
    def __init__(
        self,
        base_model: str = TEXT_BISON_MODEL,
        n_update_steps: int = 1,
        default_label: str = "Random",
    ):
        """
        Tunable Vertex-based text classifier.

        Parameters
        ----------
        base_model : str, optional
            base model to use.
        n_update_steps : int, optional
            number of epochs, by default 1
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        """
        self._set_hyperparameters(base_model=base_model, n_update_steps=n_update_steps)
        super().__init__(
            model=None,
            default_label=default_label,
        )
