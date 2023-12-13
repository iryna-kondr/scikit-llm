from skllm.llm.gpt.mixin import (
    GPTClassifierMixin as _GPTClassifierMixin,
    GPTTunableMixin as _GPTTunableMixin,
)
from skllm.models._base.classifier import (
    BaseTunableClassifier as _BaseTunableClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from typing import Optional


class _TunableClassifier(_BaseTunableClassifier, _GPTClassifierMixin, _GPTTunableMixin):
    pass


class GPTClassifier(_TunableClassifier, _SingleLabelMixin):
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        default_label: Optional[str] = "Random",
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
        prompt_template: Optional[str] = None,
    ):
        super().__init__(
            model=None, default_label=default_label, prompt_template=prompt_template
        )
        self._set_keys(key, org)
        self._set_hyperparameters(base_model, n_epochs, custom_suffix)


class MultiLabelGPTClassifier(_TunableClassifier, _MultiLabelMixin):
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        default_label: Optional[str] = "Random",
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
        prompt_template: Optional[str] = None,
        max_labels: Optional[int] = 5,
    ):
        super().__init__(
            model=None,
            default_label=default_label,
            prompt_template=prompt_template,
            max_labels=max_labels,
        )
        self._set_keys(key, org)
        self._set_hyperparameters(base_model, n_epochs, custom_suffix)
