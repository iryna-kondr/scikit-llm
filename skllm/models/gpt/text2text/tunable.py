from skllm.llm.gpt.mixin import (
    GPTTunableMixin as _GPTTunableMixin,
    GPTTextCompletionMixin as _GPTTextCompletionMixin,
)
from skllm.models._base.text2text import (
    BaseTunableText2TextModel as _BaseTunableText2TextModel,
)
from typing import Optional


class TunableGPTText2Text(
    _BaseTunableText2TextModel, _GPTTextCompletionMixin, _GPTTunableMixin
):
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
    ):
        self.model = None
        self._set_keys(key, org)
        self._set_hyperparameters(base_model, n_epochs, custom_suffix)
