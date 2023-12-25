from skllm.llm.vertex.mixin import (
    VertexTunableMixin as _VertexTunableMixin,
    VertexTextCompletionMixin as _VertexTextCompletionMixin,
)
from skllm.models._base.text2text import (
    BaseTunableText2TextModel as _BaseTunableText2TextModel,
)


class TunableVertexText2Text(
    _BaseTunableText2TextModel, _VertexTextCompletionMixin, _VertexTunableMixin
):
    def __init__(
        self,
        base_model: str = "text-bison@002",
        n_update_steps: int = 1,
    ):
        """
        Tunable Vertex-based text-to-text model.

        Parameters
        ----------
        base_model : str, optional
            base model to use, by default "text-bison@002"
        n_update_steps : int, optional
            number of epochs, by default 1
        """
        self.model = None
        self._set_hyperparameters(base_model=base_model, n_update_steps=n_update_steps)
